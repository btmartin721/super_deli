"""Execute unchanged baselines and retain every fitted candidate without refitting."""

import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

from super_deli.benchmarking.neural_study import write_dataset
from super_deli.benchmarking.unknown_k_config import ARMS, CONDITIONS
from super_deli.benchmarking.unknown_k_data import generate, model_config
from super_deli.io.config import write_json


@contextmanager
def retain_gmm_candidates(out):
    original = GaussianMixture.fit
    was_owned = "fit" in GaussianMixture.__dict__
    arrays, records = {}, []

    def fit(estimator, x, y=None):
        k = estimator.n_components
        record = {"k": k, "n_init": estimator.n_init, "random_state": estimator.random_state}
        records.append(record)
        arrays["scores"] = np.array(x, copy=True)
        try:
            result = original(estimator, x, y)
            probabilities = estimator.predict_proba(x)
            for key, value in {
                "weights": estimator.weights_,
                "means": estimator.means_,
                "covariances": estimator.covariances_,
                "membership": probabilities,
                "labels": probabilities.argmax(1),
            }.items():
                arrays[f"k{k}_{key}"] = value.copy()
            record.update(
                status="complete",
                converged=bool(estimator.converged_),
                iterations=int(estimator.n_iter_),
                bic=float(estimator.bic(x)),
            )
            return result
        except Exception as exc:
            record.update(status="failed", error=str(exc))
            raise

    GaussianMixture.fit = fit
    try:
        yield
    finally:
        if was_owned:
            GaussianMixture.fit = original
        else:
            del GaussianMixture.fit
        # Instrumentation is worker-local, read-only after fit, and never adds a fit.
        out = Path(out)
        out.mkdir(exist_ok=True)
        np.savez_compressed(out / "candidate-trace.npz", **arrays)
        write_json(records, out / "candidate-trace.json")


def run_arm(dataset, raw, out, arm, kind):
    from super_deli.workflows.tabular_gmm import run_tabular_gmm
    from super_deli.workflows.tabular_hdbscan import run_tabular_hdbscan
    from super_deli.workflows.tabular_spectral import run_tabular_spectral

    start = time.monotonic()
    try:
        kwargs = {"base_dir": raw, "output_dir": out, "model_config": model_config(arm)}
        if arm == "gmm":
            with retain_gmm_candidates(out):
                summary = run_tabular_gmm(dataset, **kwargs)
        elif arm == "density":
            summary = run_tabular_hdbscan(dataset, **kwargs)
        else:
            summary = run_tabular_spectral(dataset, **kwargs)
        record = {
            "status": summary["status"],
            "selected_k": summary["selected_k"],
            "reason": summary.get("reason"),
            "solver_called": summary.get("solver_called", False),
        }
        if arm in {"gmm", "density"}:
            record["status"] = "available"
    except Exception as exc:
        record = {"status": "failed", "selected_k": None, "error": str(exc)}
    return {"arm": arm, "kind": kind, "seconds": time.monotonic() - start, **record}


def run_dataset(args):
    root, task = args
    out = Path(root) / task["key"]
    out.mkdir()
    write_json(task, out / "task.json")
    data = generate(task)
    np.savez_compressed(out / "generating-reference.npz", **data)
    records = []
    for condition in CONDITIONS:
        raw = out / condition / "raw"
        raw.parent.mkdir()
        dataset = write_dataset(data[condition], raw)
        for arm in ARMS:
            record = run_arm(dataset, raw, raw.parent / arm, arm, "point")
            records.append({"condition": condition, **record})
            if task["replay"]:
                replay = run_arm(dataset, raw, raw.parent / (arm + "_replay"), arm, "replay")
                records.append({"condition": condition, **replay})
    write_json(records, out / "fits.json")
    return {"key": task["key"], "n": task["n"], "records": records}


def probe(out, n, target):
    from threadpoolctl import threadpool_limits

    from super_deli.benchmarking.neural_study import FEATURES
    from super_deli.models.spectral_config import SpectralConfig
    from super_deli.models.spectral_graph import embedding, graph

    out = Path(out)
    out.mkdir(parents=True)
    x = np.random.default_rng([93431, n]).normal(size=(n, 8))
    x = (x - x.mean(0)) / x.std(0)
    ids = [f"probe_{i:04d}" for i in range(n)]
    config = SpectralConfig(True, FEATURES, 15)
    start = time.monotonic()
    record = {"n": n, "target": target, "engineering_only": True}
    arrays = {"scores": x}
    try:
        with threadpool_limits(limits=1):
            affinity, values, info = graph(x, ids, config)
            arrays.update(values)
            if info["components"] != 1:
                raise ValueError("Disconnected timing fixture; no replacement")
            evidence, diagnostics = embedding(affinity, arrays, target, 71323)
            arrays.update(evidence)
            record.update(status="complete", **info, **diagnostics)
    except Exception as exc:
        record.update(status="failed", error=str(exc))
    record["seconds"] = time.monotonic() - start
    np.savez_compressed(out / "arrays.npz", **arrays)
    write_json(record, out / "result.json")
    return record
