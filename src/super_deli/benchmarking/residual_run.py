"""Independent-cohort point workflows, with live numerical checks and no resampling."""

import os
import socket
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_calibration_run import run_arm as native_arm
from super_deli.benchmarking.neural_false_split_state import atomic_json
from super_deli.benchmarking.residual_config import arms, model_for
from super_deli.benchmarking.residual_data import generate, write_cohort, write_labels
from super_deli.benchmarking.residual_evaluation import (
    aligned_matrix,
    classification_metrics,
    confidence_metrics,
    discovery_metrics,
    encode_matrix,
    frozen_membership,
)
from super_deli.benchmarking.residual_observer import ResidualObserver
from super_deli.benchmarking.view_weight_observer import NumericalAuditFailure
from super_deli.io.config import read_json, write_json
from super_deli.io.tabular import ingest_tabular
from super_deli.models.matched_preprocessing import prepare_views
from super_deli.models.neural_calibration_config import NeuralCalibrationConfig
from super_deli.workflows.residual_reference import (
    load_residual_reference_model,
    train_matched_residual,
)


def head_files(out, arm):
    return out / ("head-evidence/0" if arm == "pca4" else "evaluations/point/gmm_shared")


def external_predictions(out, cohorts, predictor, *, supervised, selected_k=None, labeled=True):
    evidence, metrics = {}, {}
    for cohort, value in cohorts.items():
        probability = predictor(value["x"])
        labels = probability.argmax(1)
        require(np.isfinite(probability).all(), "Finite external predictions")
        np.testing.assert_allclose(probability.sum(1), 1, atol=1e-10)
        evidence[cohort + "_probabilities"] = probability
        evidence[cohort + "_labels"] = labels
        if cohort == "unfamiliar":
            metrics[cohort] = confidence_metrics(probability)
            if not supervised:
                metrics[cohort]["mean_maximum_component_membership"] = metrics[cohort].pop(
                    "mean_maximum_reference_confidence"
                )
                metrics[cohort]["scope"] = (
                    "Conditional mixture membership; "
                    "not reference classification or novelty detection"
                )
        elif supervised:
            metrics[cohort] = classification_metrics(value["data"]["truth"], probability)
        else:
            metrics[cohort] = discovery_metrics(
                value["data"].get("truth"), labels, selected_k, labeled=labeled
            )
    np.savez_compressed(out / "external-predictions.npz", **evidence)
    return metrics


def supervised_arm(arm, dataset, raw, label_path, train_data, cohorts, out):
    started = time.monotonic()
    out.mkdir()
    residual = "residual_" in arm
    seed = int(arm.rsplit("_", 1)[1]) if residual else 0
    config = model_for("residual", seed)
    row = dict(
        arm=arm,
        task="reference_assignment",
        status="failed",
        attempted=1,
        neural_fits=int(residual),
        failed_points=1,
        numerical_guard_failed=False,
    )
    observer = ResidualObserver()
    try:
        if residual:
            summary = train_matched_residual(
                dataset,
                labels_path=label_path,
                model_config=config,
                base_dir=raw,
                output_dir=out / "run",
                _observer=observer,
            )
            model = load_residual_reference_model(out / "run")
            audited = observer.finish(model.arrays)
            np.savez_compressed(out / "optimizer-checks.npz", **audited)
            write_json(
                dict(
                    updates_checked=500,
                    maximum_errors=np.max(audited["errors"], axis=0).tolist(),
                    gradient_moment_parameter_tolerances=[1e-8, 1e-10],
                    loss_probability_tolerances=[1e-10, 1e-10],
                ),
                out / "optimizer-checks.json",
            )

            def predictor(x):
                return model.predict_matrix(x)["probabilities"]

            row["parameter_count"] = summary["training"]["parameter_count"]
        else:
            from super_deli.io.reference_labels import (
                read_reference_labels,
                validate_reference_labels,
            )

            loaded = ingest_tabular(
                dataset,
                base_dir=raw,
                output_dir=out / "ingestion",
                modality_names=["morphology", "ecology"],
            )
            x, units = aligned_matrix(loaded.tables, config)
            _, y, counts = validate_reference_labels(units, read_reference_labels(label_path))
            z, prep = prepare_views(x, 4)
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, tol=1e-9)
            with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(z, y)
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                raise ValueError("Reference logistic iteration limit; no replacement")
            saved = {
                **prep,
                "coefficients": model.coef_,
                "intercept": model.intercept_,
                "classes": model.classes_,
                "iterations": model.n_iter_,
            }
            np.savez_compressed(out / "model.npz", **saved)
            write_json(
                dict(
                    config=config.to_dict(),
                    class_counts=counts,
                    C=1,
                    solver="lbfgs",
                    max_iter=2000,
                    tol=1e-9,
                ),
                out / "model-config.json",
            )

            def predictor(x):
                kept = prep["kept_indices"]
                z = (
                    (x[:, kept] - prep["means"][kept])
                    / prep["scales"][kept]
                    * prep["column_weights"]
                )
                return model.predict_proba(z)

            row["parameter_count"] = int(model.coef_.size + model.intercept_.size)
        row.update(
            status="available",
            failed_points=0,
            metrics=external_predictions(out, cohorts, predictor, supervised=True),
        )
    except Exception as exc:
        row["error"] = str(exc)
        cause = exc
        while cause is not None:
            row["numerical_guard_failed"] |= isinstance(cause, NumericalAuditFailure)
            cause = cause.__cause__
        if observer.errors and not (out / "optimizer-checks.npz").exists():
            np.savez_compressed(
                out / "failed-optimizer-checks.npz",
                errors=np.asarray(observer.errors),
                step_sha256=np.asarray(observer.hashes),
            )
    row["seconds"] = time.monotonic() - started
    row["completed_optimizer_updates"] = len(observer.errors)
    write_json(row, out / "execution.json")
    return row


def unsupervised_arm(arm, dataset, raw, train_data, cohorts, out, *, labeled):
    started = time.monotonic()
    kind = "pca4" if arm == "pca4" else "neural"
    seed = 0 if kind == "pca4" else int(arm.rsplit("_", 1)[1])
    policy = NeuralCalibrationConfig(True, model_for(kind, seed), n_simulations=2)
    row = native_arm(dataset, raw, out, policy, kind, point_only=True)
    row.update(arm=arm, task="unsupervised_discovery")
    if row["status"] == "available":
        try:
            point = arrays(out / "run/point.npz")
            folder = head_files(out, kind)
            head, metadata = arrays(folder / "fit.npz"), read_json(folder / "execution.json")

            def predictor(x):
                return frozen_membership(encode_matrix(x, point, kind), head, metadata)

            row["selected_k"] = metadata["selected_k"]
            row["training_metrics"] = discovery_metrics(
                train_data.get("truth"), point["labels"], row["selected_k"], labeled=labeled
            )
            row["metrics"] = external_predictions(
                out,
                cohorts,
                predictor,
                supervised=False,
                selected_k=row["selected_k"],
                labeled=labeled,
            )
        except Exception as exc:
            row.update(status="failed", failed_points=1, error=str(exc))
    row["seconds"] = time.monotonic() - started
    write_json(row, out / "result.json")
    return row


def run_fresh(job, out):
    task = job["task"]
    data = generate(task, "train")
    np.savez_compressed(out / "generating-reference.npz", **data)
    dataset, units = write_cohort(
        data, out / "raw", tag=Path(job["phase_root"]).name + "_" + task["key"] + "_train"
    )
    if task["labeled"]:
        write_labels(units, data["truth"], out / "references.csv")
    if task["negative"]:
        permuted = np.random.default_rng([*task["seed_path"], 99]).permutation(data["truth"])
        write_labels(units, permuted, out / "permuted-references.csv")
    cohorts = {}
    names = (
        ["iid"]
        + (["shifted"] if task["case"] == "confounded_size" else [])
        + (["unfamiliar"] if task["labeled"] else [])
    )
    for name in names:
        directory = out / name
        directory.mkdir()
        values = generate(task, name)
        np.savez_compressed(directory / "generating-reference.npz", **values)
        ds, external_ids = write_cohort(
            values,
            directory / "raw",
            tag=Path(job["phase_root"]).name + "_" + task["key"] + "_" + name,
        )
        loaded = ingest_tabular(
            ds,
            base_dir=directory / "raw",
            output_dir=directory / "ingestion",
            modality_names=["morphology", "ecology"],
        )
        x, aligned = aligned_matrix(loaded.tables, model_for("residual", 0))
        require(
            aligned == external_ids and not set(units) & set(external_ids),
            "Disjoint ID-bound external cohort",
        )
        np.testing.assert_array_equal(x, np.column_stack((values["morphology"], values["ecology"])))
        cohorts[name] = dict(data=values, x=x)
    rows = []
    for arm in arms(task):
        require(not Path(job["stop_file"]).exists(), "Earlier numerical guard")
        if arm == "pca4" or arm.startswith("neural_"):
            row = unsupervised_arm(
                arm, dataset, out / "raw", data, cohorts, out / arm, labeled=task["labeled"]
            )
        else:
            label_path = out / (
                "permuted-references.csv" if arm.startswith("permuted_") else "references.csv"
            )
            row = supervised_arm(arm, dataset, out / "raw", label_path, data, cohorts, out / arm)
        rows.append(row)
        if row["numerical_guard_failed"]:
            atomic_json(
                Path(job["stop_file"]),
                dict(reason="Actual residual/nonlinear numerical guard", key=task["key"], arm=arm),
            )
            raise NumericalAuditFailure(row["error"])
    return rows


def run_replay(job, out):
    entry = job["entry"]
    source = Path(job["archive_root"]) / entry["path"]
    data = arrays(source / "reference.npz")
    ds, _ = write_cohort(data, out / "raw", tag=job["task"]["key"])
    arm = entry["arm"]
    row = unsupervised_arm(
        "pca4" if arm == "pca4" else "neural_0",
        ds,
        out / "raw",
        data,
        {},
        out / "point",
        labeled=True,
    )
    require(
        row["status"] == "available" and not row["numerical_guard_failed"],
        "Complete legacy point replay",
    )
    actual, expected = arrays(out / "point/run/point.npz"), arrays(source / (arm + "-point.npz"))
    require(actual.keys() == expected.keys(), "Every source point array retained")
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])
    head = head_files(out / "point", arm)
    for key, value in arrays(source / (arm + "-head.npz")).items():
        np.testing.assert_array_equal(arrays(head / "fit.npz")[key], value)
    np.savez_compressed(out / "generating-reference.npz", **data)
    write_json(dict(passed=True, point_arrays=len(actual)), out / "replay-parity.json")
    return [row]


def run_unit(job):
    started = time.monotonic()
    phase = Path(job["phase_root"])
    task, out = job["task"], phase / job["task"]["key"]
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
        rows = run_replay(job, out) if job["kind"] == "replay" else run_fresh(job, out)
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
