"""Paired frozen classifiers and shared support with no evaluation-label fitting."""

import os
import socket
import time
from pathlib import Path

import numpy as np

from super_deli.benchmarking.abstention_config import ARMS, POLICIES, cohort_names
from super_deli.benchmarking.abstention_data import generate
from super_deli.benchmarking.abstention_metrics import metrics, target, unscreened
from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_false_split_state import atomic_json
from super_deli.benchmarking.residual_data import write_cohort, write_labels
from super_deli.benchmarking.residual_run import supervised_arm
from super_deli.io.config import write_json
from super_deli.models.matched_preprocessing import prepare_views
from super_deli.models.reference_abstention import (
    calibrate_distance,
    calibrate_probability,
    decide,
    distances,
    fit_support,
    transform,
)
from super_deli.models.residual_network import probabilities as softmax
from super_deli.workflows.residual_reference import load_residual_reference_model


def logistic_probability(x, a):
    return softmax(transform(x, a) @ a["coefficients"].T + a["intercept"])


def run_fresh(job, out):
    task = job["task"]
    cohorts = {}
    for name in cohort_names(task):
        data = generate(task, name)
        folder = out if name == "train" else out / name
        if name != "train":
            folder.mkdir()
        np.savez_compressed(folder / "generating-reference.npz", **data)
        ds, ids = write_cohort(
            data, folder / "raw", tag=f"{Path(job['phase_root']).name}_{task['key']}_{name}"
        )
        if name in ("train", "calibration"):
            write_labels(ids, data["truth"], folder / "references.csv")
        cohorts[name] = dict(
            data=data,
            x=np.column_stack((data["morphology"], data["ecology"])),
            dataset=ds,
            ids=ids,
            folder=folder,
        )
    train, cal = cohorts.pop("train"), cohorts.pop("calibration")
    _, prep = prepare_views(train["x"], 4)
    support_error = None
    try:
        support = fit_support(train["x"], train["data"]["truth"], prep, 4)
        shared = calibrate_distance(support, cal["x"], cal["data"]["truth"])
        np.savez_compressed(out / "shared-support.npz", **shared)
        distance_scores = {name: distances(v["x"], support) for name, v in cohorts.items()}
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        support_error = str(exc)
        write_json(dict(status="unavailable", error=support_error), out / "support-failure.json")
    rows = []
    for arm in ARMS:
        require(not Path(job["stop_file"]).exists(), "Earlier numerical guard")
        row = supervised_arm(
            arm, train["dataset"], out / "raw", out / "references.csv", train["data"], {}, out / arm
        )
        if row["numerical_guard_failed"]:
            atomic_json(
                Path(job["stop_file"]),
                dict(reason="Unchanged residual actual-update guard", key=task["key"], arm=arm),
            )
            raise ValueError(row["error"])
        if row["status"] == "available":
            try:
                if support_error is not None:
                    raise ValueError("Unavailable reference support: " + support_error)
                if arm == "logistic":
                    a = arrays(out / arm / "model.npz")

                    def predictor(x, parameters=a):
                        return logistic_probability(x, parameters)
                else:
                    model = load_residual_reference_model(out / arm / "run")
                    a = model.arrays

                    def predictor(x, classifier=model):
                        return classifier.predict_matrix(x)["probabilities"]

                for k in prep:
                    np.testing.assert_array_equal(a[k], prep[k])
                fitted = calibrate_probability(shared, predictor(cal["x"]))
                np.savez_compressed(out / arm / "calibration.npz", **fitted)
                predictions = {"calibration_probabilities": fitted["calibration_probabilities"]}
                row["metrics"] = {p: {} for p in POLICIES}
                for name, v in cohorts.items():
                    p = predictor(v["x"])
                    predictions[name + "_probabilities"] = p
                    predictions[name + "_distances"] = distance_scores[name]
                    for policy in POLICIES:
                        decision = (
                            unscreened(p)
                            if policy == "unscreened"
                            else decide(
                                p, distance_scores[name], fitted[policy + "_thresholds"], policy
                            )
                        )
                        for key in ("accepted", "accepted_indices", "decisions", "compatible"):
                            if key in decision:
                                predictions[name + "_" + policy + "_" + key] = decision[key]
                        row["metrics"][policy][name] = metrics(
                            v["data"]["truth"], decision, known=name in ("iid", "shifted")
                        )
                row["joint_targets"] = {p: target(row["metrics"][p]) for p in POLICIES}
                np.savez_compressed(out / arm / "policy-predictions.npz", **predictions)
            except Exception as exc:
                row.update(status="failed", failed_points=1, error=str(exc))
        write_json(row, out / arm / "policy-result.json")
        rows.append(row)
    return rows


def replay_predictions(entry, source):
    old = Path(source) / entry["path"]
    folder = old / entry["arm"]
    residual = entry["arm"] != "logistic"
    a = arrays(folder / ("run/model.npz" if residual else "model.npz"))
    model = load_residual_reference_model(folder / "run") if residual else None
    expected = arrays(folder / "external-predictions.npz")
    result = {}
    for cohort in ("iid", "unfamiliar"):
        data = arrays(old / cohort / "generating-reference.npz")
        x = np.column_stack((data["morphology"], data["ecology"]))
        p = model.predict_matrix(x)["probabilities"] if residual else logistic_probability(x, a)
        np.testing.assert_array_equal(p, expected[cohort + "_probabilities"])
        np.testing.assert_array_equal(p.argmax(1), expected[cohort + "_labels"])
        result[cohort + "_probabilities"] = p
    return result


def run_unit(job):
    started = time.monotonic()
    phase, task = Path(job["phase_root"]), job["task"]
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
        if job["kind"] == "replay":
            predicted = replay_predictions(job["entry"], job["archive_root"])
            np.savez_compressed(out / "replay-predictions.npz", **predicted)
            rows = [
                dict(
                    arm=job["entry"]["arm"],
                    status="available",
                    attempted=1,
                    neural_fits=0,
                    failed_points=0,
                    replay_cohorts=2,
                )
            ]
            write_json(dict(passed=True, cohorts=2, new_training=False), out / "replay-parity.json")
        else:
            rows = run_fresh(job, out)
        write_json(rows, out / "fits.json")
        return dict(
            key=task["key"],
            seconds=time.monotonic() - started,
            attempted=sum(r["attempted"] for r in rows),
            neural_fits=sum(r["neural_fits"] for r in rows),
            failed_points=sum(r["failed_points"] for r in rows),
        )
    finally:
        atomic_json(
            marker,
            {**state, "status": "finished", "elapsed": time.monotonic() - started},
            replace=True,
        )
