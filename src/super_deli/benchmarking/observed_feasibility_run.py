"""Fresh diagnostic cohorts and fixed native forests; original heads never train."""

import os
import socket
import time
from pathlib import Path

import numpy as np

from super_deli.benchmarking.abstention_config import ARMS
from super_deli.benchmarking.abstention_metrics import metrics, target, unscreened
from super_deli.benchmarking.abstention_run import logistic_probability
from super_deli.benchmarking.local_reference_run import replay as old_replay
from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_false_split_state import atomic_json
from super_deli.benchmarking.observed_feasibility_config import roles
from super_deli.benchmarking.observed_feasibility_data import generate_role, measured, role_ids
from super_deli.benchmarking.observed_feasibility_tree import (
    at_threshold,
    choose_threshold,
    fit_detector,
    native_stages,
    threshold_counts,
)
from super_deli.io.config import read_json, write_json
from super_deli.models.local_reference import local_distances
from super_deli.models.reference_abstention import decide, distances
from super_deli.workflows.residual_reference import load_residual_reference_model


def original_predictors(base):
    result = {}
    for arm in ARMS:
        if arm == "logistic":
            parameters = arrays(base / arm / "model.npz")

            def predictor(x, a=parameters):
                return logistic_probability(x, a)
        else:
            model = load_residual_reference_model(base / arm / "run")

            def predictor(x, m=model):
                return m.predict_matrix(x)["probabilities"]

        result[arm] = predictor
    return result


def detector_fit(cohorts, out):
    x = np.concatenate([measured(cohorts["train_" + g]) for g in ("known", "near", "far")])
    y = np.concatenate(
        [
            np.full(len(cohorts["train_" + g]["truth"]), int(g == "known"), dtype=int)
            for g in ("known", "near", "far")
        ]
    )
    model, forest = fit_detector(x, y)
    np.savez_compressed(out / "forest.npz", **forest)
    scores = {}
    for name, d in cohorts.items():
        stages, value = native_stages(model, measured(d))
        scores[name + "_stages"] = stages
        scores[name + "_known_scores"] = value

    # sklearn 1.6 binary train_score_ is twice the native log loss.
    # Do not recompute it from clipped prediction probabilities.
    scores["native_training_log_loss"] = 0.5 * model.train_score_.copy()
    np.savez_compressed(out / "native-scores.npz", **scores)
    return scores


def curves(score, probability, truth, known):
    value = threshold_counts(score, probability.argmax(1) == truth if known else None)
    if known:
        value["raw_accuracy"] = float(np.mean(probability.argmax(1) == truth))
        value["per_class"] = [
            threshold_counts(score[truth == c], (probability.argmax(1) == truth)[truth == c])
            for c in range(4)
        ]
    return value


def selected_metrics(curve, index):
    value = at_threshold(curve, index)
    if "per_class" in curve:
        value["raw_accuracy"] = curve["raw_accuracy"]
        value["per_class"] = [
            dict(reference=c, **at_threshold(v, index)) for c, v in enumerate(curve["per_class"])
        ]
    return value


def evaluate(base, cohorts, scores, out):
    predictors = original_predictors(base)
    shared = arrays(base / "shared-support.npz")
    local = arrays(base / "local-support.npz")
    geometry = {}
    for name, d in cohorts.items():
        if name.startswith("test_"):
            x = measured(d)
            geometry[name] = (distances(x, shared), local_distances(x, shared, local))
    stored = {}
    rows = []
    for arm, predict in predictors.items():
        fitted = arrays(base / arm / "calibration.npz")
        calibration = {}
        test = {}
        original = {p: {} for p in ("unscreened", "probability", "distance", "local")}
        for name, d in cohorts.items():
            if name.startswith("train_"):
                continue
            p = predict(measured(d))
            stored[arm + "_" + name + "_probabilities"] = p
            known = name in ("calibration_known", "test_iid", "test_shifted")
            c = curves(scores[name + "_known_scores"], p, d["truth"], known)
            if name.startswith("calibration_"):
                calibration[
                    "iid" if name.endswith("known") else name.removeprefix("calibration_")
                ] = c
                continue
            key = name.removeprefix("test_")
            test[key] = c
            dist, loc = geometry[name]
            for policy in original:
                decision = (
                    unscreened(p)
                    if policy == "unscreened"
                    else decide(
                        p,
                        loc if policy == "local" else dist,
                        local["thresholds"]
                        if policy == "local"
                        else fitted[policy + "_thresholds"],
                        "distance" if policy == "local" else policy,
                    )
                )
                original[policy][key] = metrics(d["truth"], decision, known=known)
                for field in ("accepted", "accepted_indices", "decisions", "compatible"):
                    if field in decision:
                        stored[arm + "_" + name + "_" + policy + "_" + field] = decision[field]
        choice = choose_threshold(calibration["iid"], calibration["near"], calibration["far"])
        fixed = (
            {c: selected_metrics(v, choice["index"]) for c, v in test.items()}
            if choice["status"] == "selected"
            else None
        )
        original["diagnostic"] = fixed
        rows.append(
            dict(
                arm=arm,
                status="available",
                selection=choice,
                calibration_curves=calibration,
                evaluation_curves=test,
                metrics=original,
                joint_targets={
                    **{p: target(v) for p, v in original.items() if p != "diagnostic"},
                    "diagnostic": target(fixed) if fixed else False,
                },
            )
        )
    np.savez_compressed(out / "reference-predictions.npz", **stored)
    write_json(rows, out / "assessment.json")
    return rows


def ordinary(job, out):
    task = job["task"]
    cohorts = {}
    identities = {}
    for role in roles(task):
        name = role["name"]
        cohorts[name] = generate_role(task, name)
        identities[name] = role_ids(task, name)
        np.savez_compressed(out / (name + ".npz"), **cohorts[name])
    write_json(identities, out / "identities.json")
    scores = detector_fit(cohorts, out)
    rows = evaluate(Path(job["archive_root"]) / job["entry"]["path"], cohorts, scores, out)
    return dict(
        detector_fits=1, trees=100, assessed_heads=len(rows), source_path=job["entry"]["path"]
    )


def detector_replay(job, out):
    original = Path(job["phase_root"]) / job["original_key"]
    task = read_json(original / "task.json")
    cohorts = {r["name"]: arrays(original / (r["name"] + ".npz")) for r in roles(task)}
    detector_fit(cohorts, out)
    for name in ("forest.npz", "native-scores.npz"):
        a, b = arrays(original / name), arrays(out / name)
        require(a.keys() == b.keys(), "Exact replay fields")
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    write_json(
        dict(
            passed=True,
            original_key=job["original_key"],
            new_cohorts=False,
            new_reference_training=False,
        ),
        out / "detector-replay.json",
    )
    return dict(detector_fits=1, trees=100, assessed_heads=0)


def frozen_replay(job, out):
    source = Path(job["archive_root"])
    entry = job["entry"]
    old_replay(entry, source, out)
    base = source / entry["path"]
    old = arrays(base / entry["arm"] / "local-predictions.npz")
    predictor = original_predictors(base)[entry["arm"]]
    shared, local = arrays(base / "shared-support.npz"), arrays(base / "local-support.npz")
    retained = {}
    for name in ("iid", "near", "far", "overlap"):
        data = arrays(base / name / "generating-reference.npz")
        x = measured(data)
        p = predictor(x)
        result = decide(p, local_distances(x, shared, local), local["thresholds"], "distance")
        for field in ("accepted", "accepted_indices", "decisions", "compatible"):
            key = name + "_" + field
            np.testing.assert_array_equal(result[field], old[key])
            retained[key] = result[field]
    np.savez_compressed(out / "local-replay.npz", **retained)
    return dict(detector_fits=0, trees=0, assessed_heads=1, replay_cohorts=5)


def _run_unit(job):
    started = time.monotonic()
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
        started=started,
        elapsed=None,
    )
    atomic_json(marker, state)
    write_json(task, out / "task.json")
    try:
        require(not Path(job["stop_file"]).exists(), "Earlier retained guard")
        value = (
            ordinary
            if job["kind"] == "fresh"
            else detector_replay
            if job["kind"] == "detector_replay"
            else frozen_replay
        )(job, out)
        write_json(dict(status="complete", kind=job["kind"], **value), out / "unit-result.json")
        return dict(
            key=task["key"],
            seconds=time.monotonic() - started,
            attempted=1,
            neural_fits=0,
            failed_points=0,
            **value,
        )
    except Exception as exc:
        write_json(dict(status="failed", error=str(exc), kind=job["kind"]), out / "failure.json")
        raise
    finally:
        atomic_json(
            marker,
            {**state, "status": "finished", "elapsed": time.monotonic() - started},
            replace=True,
        )


def run_unit(job):
    from super_deli.benchmarking.observed_feasibility_runtime import scientific_runtime

    with scientific_runtime():
        from importlib import import_module

        from threadpoolctl import threadpool_limits

        import_module("sklearn.ensemble")
        with threadpool_limits(limits=1):
            return _run_unit(job)
