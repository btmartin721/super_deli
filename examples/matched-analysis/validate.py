"""Bounded orchestration validation; synthetic observations are not species evidence.

Run preflight before main. Every native replay is an engineering parity check,
excluded from performance denominators. No threshold/model selection is added.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import time
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

import super_deli
from super_deli.benchmarking.matched_data import generate, write_dataset
from super_deli.benchmarking.matched_reference import reference_data
from super_deli.benchmarking.matched_steps import digest_arrays
from super_deli.benchmarking.neural_analysis import arrays
from super_deli.benchmarking.separate_gmm_observer import evaluation_path, retain_points
from super_deli.benchmarking.view_weight_audit import audit_trajectory
from super_deli.benchmarking.view_weight_observer import retain_training
from super_deli.io.config import load_config, read_json, write_json
from super_deli.models.matched_analysis_config import MatchedAnalysisConfig
from super_deli.models.separate_neural import unpack_branch
from super_deli.workflows.matched_analysis import run_matched_analysis
from super_deli.workflows.matched_analysis_adapters import run_matched_arm

CASES = ("gaussian_independent", "four_complementary", "moons")
CAP = 4 * 1024**3
FREE = 8 * 1024**3


def sha(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def retained(root):
    return sum((Path(d) / n).lstat().st_size for d, _, files in os.walk(root) for n in files)


def schedule(phase):
    return [
        dict(
            key=f"{case}_n{n}_{r:02d}",
            case=case,
            n=n,
            replicate=r,
            draws=1 if phase == "preflight" else 5 if r == 0 else 0,
            **{
                name: [20261114, 770, int(phase == "main"), i, n, r, j]
                for j, name in enumerate(
                    (
                        "data_seed",
                        "permutation_seed",
                        "noise_seed",
                        "ecology_order_seed",
                        "resampling_seed",
                    )
                )
            },
        )
        for i, case in enumerate(CASES)
        for n in (64, 128)
        for r in range(1 if phase == "preflight" else 3)
    ]


def declaration():
    package = Path(super_deli.__file__).parent
    import numpy._core._multiarray_umath as multiarray
    import scipy.linalg._fblas as blas
    import torch

    native_paths = (
        Path(blas.__file__),
        Path(multiarray.__file__),
        Path(torch.__file__).parent
        / "lib"
        / ("libtorch_cpu.dylib" if platform.system() == "Darwin" else "libtorch_cpu.so"),
    )
    native = {str(p): sha(p) for p in native_paths}
    return dict(
        version=super_deli.__version__,
        python=platform.python_version(),
        software={
            name: version(name)
            for name in ("numpy", "scipy", "scikit-learn", "torch", "threadpoolctl")
        },
        native_sha256=native,
        source_sha256={str(p.relative_to(package)): sha(p) for p in sorted(package.rglob("*.py"))},
        validation_sha256=sha(Path(__file__)),
        analysis=read_json(Path(__file__).with_name("analysis.json")),
        schedules={phase: schedule(phase) for phase in ("preflight", "main")},
        inference="not_computed",
        purpose="Native parity and integration coverage; not a new comparative power study",
    )


def models(task):
    cfg = MatchedAnalysisConfig.from_dict(read_json(Path(__file__).with_name("analysis.json")))
    seed = int(np.random.SeedSequence(task["resampling_seed"]).generate_state(1)[0])
    return replace(
        cfg,
        arms=tuple(
            replace(
                a,
                model=replace(
                    a.model,
                    **(
                        dict(n_bootstrap=task["draws"])
                        if a.family == "pca_gmm"
                        else dict(n_subsamples=task["draws"], subsampling_seed=seed)
                    ),
                ),
            )
            for a in cfg.arms
        ),
    )


def selected_density(path, family):
    a = arrays(path / ("mixture.npz" if family == "pca_gmm" else "point.npz"))
    scores = a["scores"]
    membership = a["membership"]
    n, d = scores.shape
    k = membership.shape[1]
    weights = a["weights"] if family == "pca_gmm" else a["mixture_weights"]
    means = a["means"] if family == "pca_gmm" else a["mixture_means"]
    cov = a["covariances"][0] if family == "pca_gmm" else a["mixture_covariance"]
    if family == "pca_gmm":
        np.testing.assert_array_equal(a["covariances"], np.repeat(cov[None], k, axis=0))
    sign, logdet = np.linalg.slogdet(cov)
    assert sign > 0
    delta = scores[:, None] - means[None]
    quadratic = np.einsum("nkd,de,nke->nk", delta, np.linalg.inv(cov), delta)
    logp = np.log(weights)[None] - 0.5 * (d * np.log(2 * np.pi) + logdet + quadratic)
    density = logsumexp(logp, axis=1)
    expected = np.exp(logp - density[:, None])
    np.testing.assert_allclose(expected, membership, rtol=1e-8, atol=1e-10)
    np.testing.assert_array_equal(a["labels"], expected.argmax(1))
    p = k * d + d * (d + 1) // 2 + k - 1
    bic = -2 * density.sum() + p * np.log(n)
    candidates = read_json(path / "candidates.json")
    chosen = [c for c in candidates if c["k"] == k][0]
    np.testing.assert_allclose(chosen["bic"], bic, rtol=0, atol=1e-6)
    eligible = [c for c in candidates if c["status"] == "eligible"]
    assert min(eligible, key=lambda c: (c["bic"], c["k"]))["k"] == k
    return int(k)


def pca_projection(data, path):
    parts = []
    for i, role in enumerate(("morphology", "ecology")):
        a = arrays(path / f"view_{i:03d}/parameters/pca.npz")
        b = arrays(path / f"view_{i:03d}/parameters/profiles.npz")
        x = data[role]
        kept = a["kept_indices"]
        z = (x[:, kept] - a["feature_means"][kept]) / a["feature_scales"][kept]
        score = (z - a["mean"]) @ a["components"].T
        np.testing.assert_allclose(score, b["profiles"], atol=1e-10)
        np.testing.assert_allclose(b["mean"], score.mean(0), atol=1e-12)
        np.testing.assert_allclose(
            b["scale"], np.sqrt(((score - score.mean(0)) ** 2).sum() / (len(x) - 1)), atol=1e-12
        )
        parts.append((score - b["mean"]) * np.sqrt(b["normalized_weight"]) / b["scale"])
    fused = np.column_stack(parts)
    f = arrays(path / "fusion.npz")
    np.testing.assert_allclose(fused, f["fused"], atol=1e-10)
    np.testing.assert_allclose(
        (fused - f["mean"]) @ f["components"].T, arrays(path / "mixture.npz")["scores"], atol=1e-10
    )


def audit_neural(trace, data, model, draws):
    events = read_json(trace / "optimizer-checks.json")
    evidence = arrays(trace / "optimizer-checks.npz")
    used = set()
    cache = {}
    updates = 0
    for index, ids in enumerate(
        [np.arange(len(data["morphology"])), *[np.asarray(d["training_indices"]) for d in draws]]
    ):
        a = arrays(evaluation_path(trace, index, "coordinates") / "point.npz")
        for role in ("morphology", "ecology"):
            b = unpack_branch(a, role)
            x = data[role][ids]
            z = (x[:, b["kept_indices"]] - b["means"][b["kept_indices"]]) / b["scales"][
                b["kept_indices"]
            ]
            matching = [
                e
                for e in events
                if e["fit"] not in used and e["input_sha256"] == digest_arrays({"input": z})
            ]
            assert matching
            event = matching[0]
            used.add(event["fit"])
            audit_trajectory(
                x, b, model.to_dict(), event, evidence, event["fit"], cache, paired=False
            )
            updates += event["updates_checked"]
    assert len(used) == len(events) == 2 * (1 + len(draws))
    return updates


def parity(left, right):
    paths = sorted(p.relative_to(left) for p in left.rglob("*") if p.suffix in (".npz", ".npy"))
    assert paths == sorted(
        p.relative_to(right) for p in right.rglob("*") if p.suffix in (".npz", ".npy")
    )
    values = 0
    for relative in paths:
        if relative.suffix == ".npz":
            a, b = arrays(left / relative), arrays(right / relative)
            assert set(a) == set(b)
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
                values += a[key].size
        else:
            a = np.load(left / relative, allow_pickle=False)
            b = np.load(right / relative, allow_pickle=False)
            np.testing.assert_array_equal(a, b)
            values += a.size
    for name in (
        "candidates.json",
        "bootstrap_replicates.json",
        "clusters.json",
        "subsamples.json",
        "cohort.json",
        "alignment.json",
        "training.json",
    ):
        if (left / name).exists():
            assert read_json(left / name) == read_json(right / name), name
    return values


def one(task, root):
    started = time.monotonic()
    out = root / task["key"]
    out.mkdir()
    data = generate(task)
    reference = reference_data(task)
    assert set(data) == set(reference)
    for key in data:
        np.testing.assert_allclose(data[key], reference[key], rtol=1e-12, atol=1e-12)
    np.savez_compressed(out / "raw.npz", **data)
    dataset = write_dataset(data, out / "data")
    cfg = models(task)
    write_json(cfg.to_dict(), out / "analysis.json")
    guard = {}
    with retain_training(out / "trace-unified", paired=False, guard_state=guard, record_times=True):
        with retain_points(out / "trace-unified"):
            summary = run_matched_analysis(
                dataset, analysis_config=cfg, base_dir=out / "data", output_dir=out / "analysis"
            )
    assert not guard.get("failed") and summary["status"] == "complete"
    frozen = load_config(out / "analysis/inputs/dataset.json")
    (out / "standalone").mkdir()
    with retain_training(
        out / "trace-standalone", paired=False, guard_state=guard, record_times=True
    ):
        with retain_points(out / "trace-standalone"):
            for arm in cfg.arms:
                run_matched_arm(arm, frozen, out / "analysis/inputs", out / "standalone" / arm.name)
    assert not guard.get("failed")
    audit_start = time.monotonic()
    updates = 0
    values = 0
    ks = {}
    labels = {}
    for arm in cfg.arms:
        native = out / "analysis/arms" / arm.name
        standalone = out / "standalone" / arm.name
        values += parity(native, standalone)
        ks[arm.name] = selected_density(native, arm.family)
        a = arrays(native / ("mixture.npz" if arm.family == "pca_gmm" else "point.npz"))
        labels[arm.name] = a["labels"]
        if arm.family == "pca_gmm":
            pca_projection(data, native)
        else:
            draws = read_json(native / "subsamples.json")
            updates += audit_neural(out / "trace-unified", data, arm.model, draws)
            updates += audit_neural(out / "trace-standalone", data, arm.model, draws)
    a, b = labels["pca"], labels["neural"]
    comparison = summary["comparisons"][0]
    np.testing.assert_allclose(comparison["adjusted_rand"], adjusted_rand_score(a, b), atol=1e-12)
    same_a = a[:, None] == a[None]
    same_b = b[:, None] == b[None]
    upper = np.triu_indices(len(a), 1)
    assert comparison["discordant_pairs"] == int((same_a[upper] != same_b[upper]).sum())
    assert comparison["joint_resolved"] == task["n"] and comparison["joint_coverage"] == 1
    assert summary["inference"] == dict(status="not_computed", p_value=None)
    assert updates == 2000 * (1 + task["draws"])
    row = dict(
        key=task["key"],
        case=task["case"],
        n=task["n"],
        replicate=task["replicate"],
        draws=task["draws"],
        passed=True,
        pipeline_fits=4 * (1 + task["draws"]),
        branch_trainings=4 * (1 + task["draws"]),
        verified_optimizer_updates=updates,
        parity_values=values,
        selected_k=ks,
        adjusted_rand=comparison["adjusted_rand"],
        seconds=time.monotonic() - started,
        audit_seconds=time.monotonic() - audit_start,
        bytes=retained(out),
    )
    write_json(row, out / "audit.json")
    return row


def run(output, phase, preflight, staging_bytes):
    assert phase in ("preflight", "main")
    assert type(staging_bytes) is int and staging_bytes >= 0
    if phase == "main":
        assert preflight is not None
    output = Path(output)
    assert not output.exists()
    output.mkdir()
    frozen = declaration()
    write_json(frozen, output / "freeze.json")
    tasks = schedule(phase)
    write_json(tasks, output / "seed-registry.json")
    if phase == "main":
        previous = read_json(Path(preflight) / "summary.json")
        assert previous["passed"] and read_json(Path(preflight) / "freeze.json") == frozen
        estimates = []
        for task in tasks:
            measured = next(
                r for r in previous["records"] if r["case"] == task["case"] and r["n"] == task["n"]
            )
            ratio = (1 + task["draws"]) / 2
            estimates.append(
                dict(seconds=measured["seconds"] * ratio, bytes=measured["bytes"] * ratio)
            )
        projected = 1.25 * sum(r["seconds"] for r in estimates)
        storage = (
            retained(Path(preflight))
            + 2 * staging_bytes
            + int(1.25 * sum(r["bytes"] for r in estimates))
            + 128 * 1024**2
        )
        gate = dict(
            passed=projected < 3600 and storage < CAP and shutil.disk_usage(output).free >= FREE,
            projected_main_audit_serial_seconds=projected,
            projected_retained_bytes=storage,
            staging_bytes_counted_twice=staging_bytes,
            cap_bytes=CAP,
            min_free_bytes=FREE,
        )
        write_json(gate, output / "gate.json")
        assert gate["passed"]
    started = time.monotonic()
    records = []
    try:
        with threadpool_limits(limits=1):
            for task in tasks:
                assert declaration() == frozen
                assert shutil.disk_usage(output).free >= FREE
                assert retained(output) + 2 * staging_bytes < CAP
                row = one(task, output)
                records.append(row)
                print(json.dumps(row), flush=True)
                assert time.monotonic() - started < (1200 if phase == "preflight" else 3600)
        fits = sum(r["pipeline_fits"] for r in records)
        updates = sum(r["verified_optimizer_updates"] for r in records)
        assert fits == (48 if phase == "preflight" else 192) and updates == (
            24000 if phase == "preflight" else 96000
        )
        summary = dict(
            passed=True,
            phase=phase,
            purpose=frozen["purpose"],
            observations=len(records),
            records=records,
            pipeline_fits=fits,
            verified_optimizer_updates=updates,
            seconds=time.monotonic() - started,
            defaults_changed=False,
            pvalues_changed=False,
            author_contact=False,
            inference="not_computed",
        )
        write_json(summary, output / "summary.json")
        return summary
    except Exception as exc:
        write_json(
            dict(passed=False, error=f"{type(exc).__name__}: {exc}", completed_records=records),
            output / "STOP.json",
        )
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--phase", choices=("preflight", "main"), required=True)
    p.add_argument("--preflight", type=Path)
    p.add_argument("--staging-bytes", type=int, default=0)
    a = p.parse_args()
    run(a.output_dir, a.phase, a.preflight, a.staging_bytes)
