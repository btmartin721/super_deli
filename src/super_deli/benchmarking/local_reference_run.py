"""Paired scoring of native fresh fits and immutable archived observations."""

import os
import socket
import time
from pathlib import Path

import numpy as np

from super_deli.benchmarking.abstention_config import cohort_names
from super_deli.benchmarking.abstention_metrics import metrics, target, unscreened
from super_deli.benchmarking.abstention_run import logistic_probability, run_fresh
from super_deli.benchmarking.local_reference_oracle import evaluate as oracle
from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_false_split_state import atomic_json
from super_deli.io.config import read_json, write_json
from super_deli.models.local_reference import (
    calibrate_local_support,
    fit_local_support,
    local_distances,
)
from super_deli.models.reference_abstention import decide
from super_deli.workflows.residual_reference import load_residual_reference_model


def data_at(root, task):
    return {
        name: arrays((root if name == "train" else root / name) / "generating-reference.npz")
        for name in cohort_names(task)
    }


def add_local(base, out, task, rows):
    cohorts = data_at(base, task)
    train, cal = cohorts["train"], cohorts["calibration"]

    def matrix(data):
        return np.column_stack((data["morphology"], data["ecology"]))

    failure = None
    try:
        metric = arrays(base / "shared-support.npz")
        support = fit_local_support(matrix(train), train["truth"], metric)
        fitted = calibrate_local_support(support, metric, matrix(cal), cal["truth"])
        np.savez_compressed(out / "local-support.npz", **fitted)
        geometry = {}
        for name, d in cohorts.items():
            if name in ("train", "calibration"):
                continue
            scores, indices, ties = local_distances(matrix(d), metric, fitted, provenance=True)
            (
                geometry[name + "_scores"],
                geometry[name + "_neighbor_indices"],
                geometry[name + "_cutoff_tie_counts"],
            ) = scores, indices, ties
        np.savez_compressed(out / "local-geometry.npz", **geometry)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        failure = str(exc)
        write_json(dict(status="unavailable", error=failure), out / "local-support-failure.json")
    for row in rows:
        arm = row["arm"]
        if row["status"] != "available" or failure is not None:
            row.update(
                local_status="failed",
                local_error=failure or row.get("error", "Classifier unavailable"),
            )
            row.setdefault("joint_targets", {})["local"] = None
            continue
        original = arrays(base / arm / "policy-predictions.npz")
        row["metrics"]["local"] = {}
        retained = {}
        for name, d in cohorts.items():
            if name in ("train", "calibration"):
                continue
            p = original[name + "_probabilities"]
            result = decide(p, geometry[name + "_scores"], fitted["thresholds"], "distance")
            row["metrics"]["local"][name] = metrics(
                d["truth"], result, known=name in ("iid", "shifted")
            )
            for key in ("accepted", "accepted_indices", "compatible", "decisions"):
                retained[name + "_" + key] = result[key]
        row["joint_targets"]["local"] = target(row["metrics"]["local"])
        row["local_status"] = "available"
        destination = out / arm
        destination.mkdir(exist_ok=True)
        np.savez_compressed(destination / "local-predictions.npz", **retained)
        write_json(
            dict(
                status="available",
                metrics=row["metrics"]["local"],
                joint_target=row["joint_targets"]["local"],
            ),
            destination / "local-result.json",
        )
    values, summary = oracle(cohorts, task)
    np.savez_compressed(out / "oracle.npz", **values)
    write_json(summary, out / "oracle-result.json")
    return rows


def replay(entry, source, out):
    base = source / entry["path"]
    arm = entry["arm"]
    saved = arrays(base / arm / "policy-predictions.npz")
    if arm == "logistic":
        a = arrays(base / arm / "model.npz")

        def predictor(x):
            return logistic_probability(x, a)
    else:
        model = load_residual_reference_model(base / arm / "run")

        def predictor(x):
            return model.predict_matrix(x)["probabilities"]

    retained = {}
    policies = {}
    fitted = arrays(base / arm / "calibration.npz")
    original_row = next(r for r in read_json(base / "fits.json") if r["arm"] == arm)
    for name in ("calibration", "iid", "near", "far", "overlap"):
        data = arrays(base / name / "generating-reference.npz")
        p = predictor(np.column_stack((data["morphology"], data["ecology"])))
        np.testing.assert_array_equal(p, saved[name + "_probabilities"])
        retained[name + "_probabilities"] = p
        if name != "calibration":
            for policy in ("unscreened", "probability", "distance"):
                result = (
                    unscreened(p)
                    if policy == "unscreened"
                    else decide(
                        p, saved[name + "_distances"], fitted[policy + "_thresholds"], policy
                    )
                )
                for key in ("accepted", "accepted_indices", "decisions", "compatible"):
                    if key in result:
                        prefix = name + "_" + policy + "_" + key
                        np.testing.assert_array_equal(result[key], saved[prefix])
                        policies[prefix] = result[key]
                require(
                    metrics(data["truth"], result, known=name == "iid")
                    == original_row["metrics"][policy][name],
                    "Exact archived policy metric replay",
                )
    np.savez_compressed(out / "replay-predictions.npz", **retained)
    np.savez_compressed(out / "replay-policies.npz", **policies)
    write_json(dict(passed=True, cohorts=5, new_training=False), out / "replay-parity.json")
    return [
        dict(
            arm=arm,
            status="available",
            attempted=1,
            neural_fits=0,
            failed_points=0,
            replay_cohorts=5,
        )
    ]


def run_unit(job):
    start = time.monotonic()
    phase = Path(job["phase_root"])
    task = job["task"]
    out = phase / task["key"]
    out.mkdir()
    marker = phase.parent / "state/active" / (phase.name + "_" + task["key"] + ".json")
    state = dict(
        pid=os.getpid(),
        host=socket.gethostname(),
        status="running",
        key=task["key"],
        started=start,
        elapsed=None,
    )
    atomic_json(marker, state)
    write_json(task, out / "task.json")
    try:
        require(not Path(job["stop_file"]).exists(), "Earlier numerical guard")
        if job["kind"] == "fresh":
            rows = run_fresh(job, out)
            rows = add_local(out, out, task, rows)
        elif job["kind"] == "archive":
            entry = job["entry"]
            base = Path(job["archive_root"]) / entry["path"]
            write_json(
                dict(path=entry["path"], source_task=entry["task"], new_training=False),
                out / "source-pointer.json",
            )
            rows = read_json(base / "fits.json")
            for row in rows:
                row.update(
                    original_neural_fits=row["neural_fits"],
                    neural_fits=0,
                    attempted=1,
                    archived=True,
                )
            rows = add_local(base, out, entry["task"], rows)
        else:
            rows = replay(job["entry"], Path(job["archive_root"]), out)
        write_json(rows, out / "fits.json")
        return dict(
            key=task["key"],
            seconds=time.monotonic() - start,
            attempted=sum(r["attempted"] for r in rows),
            neural_fits=sum(r["neural_fits"] for r in rows),
            failed_points=sum(
                r["status"] != "available" or r.get("local_status") == "failed" for r in rows
            ),
        )
    finally:
        atomic_json(
            marker,
            {**state, "status": "finished", "elapsed": time.monotonic() - start},
            replace=True,
        )
