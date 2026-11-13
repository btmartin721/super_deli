"""Known-only refits: exclusion precedes preprocessing, heads, support and calibration."""

import os
import socket
import time
from pathlib import Path

import numpy as np

from super_deli.benchmarking.abstention_metrics import metrics, unscreened
from super_deli.benchmarking.abstention_run import logistic_probability
from super_deli.benchmarking.known_reference_config import ARMS, POLICIES, roles
from super_deli.benchmarking.known_reference_data import fold_data, generate, matrix
from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_false_split_state import atomic_json
from super_deli.benchmarking.residual_data import write_cohort, write_labels
from super_deli.benchmarking.residual_run import supervised_arm
from super_deli.io.config import write_json
from super_deli.models.local_reference import (
    calibrate_local_support,
    fit_local_support,
    local_distances,
)
from super_deli.models.matched_preprocessing import prepare_views
from super_deli.models.reference_abstention import (
    calibrate_distance,
    calibrate_probability,
    decide,
    distances,
    fit_support,
)
from super_deli.models.reference_density import (
    calibrate_density_support,
    density_scores,
    fit_density_support,
)
from super_deli.workflows.residual_reference import load_residual_reference_model


def joint_target(cohorts):
    known = cohorts["iid"]
    return bool(
        known["accepted_fraction"] >= 0.8
        and known["accepted_accuracy"] is not None
        and known["accepted_accuracy"] >= 0.9
        and all(cohorts[k]["accepted_fraction"] <= 0.1 for k in ("heldout", "near", "far"))
    )


def run_fold(parent, heldout, out, tag, stop_file):
    out.mkdir()
    fold = fold_data(parent, heldout)
    manifest = {k: v for k, v in fold.items() if k not in ("training", "calibration", "cohorts")}
    manifest["evaluations"] = {
        k: {q: v[q] for q in ("parent_role", "parent_rows", "known")}
        for k, v in fold["cohorts"].items()
    }
    write_json(manifest, out / "exclusion-manifest.json")
    data = fold["training"]
    cal = fold["calibration"]
    ds, ids = write_cohort(data, out / "training", tag=tag + "_train")
    cs, cids = write_cohort(cal, out / "calibration", tag=tag + "_calibration")
    write_labels(ids, data["truth"], out / "training-labels.csv")
    write_labels(cids, cal["truth"], out / "calibration-labels.csv")
    write_json(dict(training=ds.to_dict(), calibration=cs.to_dict()), out / "input-configs.json")
    x, xc = matrix(data), matrix(cal)
    _, prep = prepare_views(x, 4)
    base = calibrate_distance(fit_support(x, data["truth"], prep, 3), xc, cal["truth"])
    local = calibrate_local_support(
        fit_local_support(x, data["truth"], base), base, xc, cal["truth"]
    )
    np.savez_compressed(out / "pooled-support.npz", **base)
    np.savez_compressed(out / "local-support.npz", **local)
    density = None
    density_error = None
    try:
        density = calibrate_density_support(
            fit_density_support(x, data["truth"], prep, 3), xc, cal["truth"]
        )
        np.savez_compressed(out / "density-support.npz", **density)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        density_error = str(exc)
        write_json(dict(status="unavailable", error=density_error), out / "density-failure.json")
    geometry = {}
    score_by_policy = {p: {} for p in ("distance", "local", "density")}
    for name, v in fold["cohorts"].items():
        score_by_policy["distance"][name] = distances(v["x"], base)
        score_by_policy["local"][name] = local_distances(v["x"], base, local)
        if density is not None:
            score_by_policy["density"][name] = density_scores(v["x"], density)
        for policy in score_by_policy:
            if name in score_by_policy[policy]:
                geometry[name + "_" + policy] = score_by_policy[policy][name]
    np.savez_compressed(out / "geometry.npz", **geometry)
    rows = []
    for arm in ARMS:
        require(not Path(stop_file).exists(), "Earlier numerical guard")
        row = supervised_arm(
            arm, ds, out / "training", out / "training-labels.csv", data, {}, out / arm
        )
        if row["numerical_guard_failed"]:
            atomic_json(
                Path(stop_file), dict(reason="Unchanged residual actual-update guard", arm=arm)
            )
            raise ValueError(row["error"])
        row["heldout"] = heldout
        row["policy_status"] = {p: "unavailable" for p in POLICIES}
        row["joint_targets"] = {p: None for p in POLICIES}
        if row["status"] == "available":
            a = arrays(out / arm / ("model.npz" if arm == "logistic" else "run/model.npz"))
            model = load_residual_reference_model(out / arm / "run") if arm != "logistic" else None

            def predict(z, model=model, a=a):
                return (
                    model.predict_matrix(z)["probabilities"]
                    if model
                    else logistic_probability(z, a)
                )

            for k in prep:
                np.testing.assert_array_equal(a[k], prep[k])
            fitted = calibrate_probability(base, predict(xc))
            np.savez_compressed(out / arm / "calibration.npz", **fitted)
            evidence = {"calibration_probabilities": fitted["calibration_probabilities"]}
            row["metrics"] = {}
            for policy in POLICIES:
                if policy == "density" and density is None:
                    row["density_error"] = density_error
                    continue
                row["policy_status"][policy] = "available"
                row["metrics"][policy] = {}
            for name, v in fold["cohorts"].items():
                p = predict(v["x"])
                evidence[name + "_probabilities"] = p
                for policy in row["metrics"]:
                    if policy == "unscreened":
                        result = unscreened(p)
                    else:
                        threshold = (
                            fitted["probability_thresholds"]
                            if policy == "probability"
                            else base["distance_thresholds"]
                            if policy == "distance"
                            else local["thresholds"]
                            if policy == "local"
                            else density["thresholds"]
                        )
                        score = (
                            score_by_policy["distance"][name]
                            if policy == "probability"
                            else score_by_policy[policy][name]
                        )
                        result = decide(
                            p,
                            score,
                            threshold,
                            "probability" if policy == "probability" else "distance",
                        )
                    for k in ("accepted", "accepted_indices", "decisions", "compatible"):
                        if k in result:
                            evidence[name + "_" + policy + "_" + k] = result[k]
                    row["metrics"][policy][name] = metrics(v["y"], result, known=v["known"])
            row["joint_targets"].update({p: joint_target(m) for p, m in row["metrics"].items()})
            np.savez_compressed(out / arm / "predictions.npz", **evidence)
            row["failed_points"] = int(density is None)
        write_json(row, out / arm / "assessment.json")
        rows.append(row)
    write_json(rows, out / "assessments.json")
    return rows


def same_arrays(left, right):
    expected = {p.relative_to(left) for p in Path(left).rglob("*.npz")}
    actual = {p.relative_to(right) for p in Path(right).rglob("*.npz")}
    require(expected == actual, "Exact fold replay numeric inventory")
    checked = 0
    for name in sorted(expected):
        a, b = arrays(Path(left) / name), arrays(Path(right) / name)
        require(a.keys() == b.keys(), "Exact replay array keys")
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
            checked += 1
    return checked


def run_unit(job):
    started = time.monotonic()
    phase = Path(job["phase_root"])
    task = job["task"]
    out = phase / task["key"]
    marker = phase.parent / "state/active" / (phase.name + "_" + task["key"] + ".json")
    state = dict(
        pid=os.getpid(),
        host=socket.gethostname(),
        status="running",
        key=task["key"],
        started=started,
        elapsed=None,
    )
    atomic_json(marker, state)
    out.mkdir()
    write_json(task, out / "task.json")
    try:
        if job["kind"] == "fresh":
            parent = {}
            (out / "parent").mkdir()
            for role, _ in roles(task):
                parent[role] = generate(task, role)
                np.savez_compressed(out / "parent" / (role + ".npz"), **parent[role])
            original = task
            heldouts = range(4)
        else:
            original = job["parent_task"]
            base = phase / original["key"]
            parent = {
                role: arrays(base / "parent" / (role + ".npz")) for role, _ in roles(original)
            }
            heldouts = [job["heldout"]]
        rows = []
        replay_arrays = 0
        for heldout in heldouts:
            folder = out / f"holdout-{heldout}"
            fitted = run_fold(
                parent,
                heldout,
                folder,
                f"{phase.name}_{original['key']}_hold{heldout}",
                job["stop_file"],
            )
            rows.extend(fitted)
            if job["kind"] != "fresh":
                replay_arrays += same_arrays(phase / original["key"] / f"holdout-{heldout}", folder)
        write_json(rows, out / "fits.json")
        if job["kind"] != "fresh":
            write_json(
                dict(passed=True, scientific_arrays=replay_arrays), out / "replay-parity.json"
            )
        return dict(
            key=task["key"],
            seconds=time.monotonic() - started,
            attempted=sum(r["attempted"] for r in rows),
            neural_fits=sum(r["neural_fits"] for r in rows),
            failed_points=sum(r["failed_points"] for r in rows),
            fold_fits=len(heldouts),
            replay_arrays=replay_arrays,
        )
    finally:
        atomic_json(
            marker,
            {**state, "status": "finished", "elapsed": time.monotonic() - started},
            replace=True,
        )
