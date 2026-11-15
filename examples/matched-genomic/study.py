"""Frozen bounded integration validation; all detailed evidence stays in output_dir."""

import argparse
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import adjusted_rand_score
from smoke import invented

from super_deli.core.schema import config_from_dict
from super_deli.io.config import read_json, write_json
from super_deli.io.matched_analysis import verify_sources
from super_deli.workflows.genomic_analysis import run_genomic_analysis
from super_deli.workflows.genomic_analysis_adapters import read_genomic_arm, run_genomic_arm

LAWS = ("common", "redundant", "complementary", "conflicting", "genomic_only", "phenotype_only")
SETTINGS = dict(
    schema_version=1,
    main_seeds=list(range(3900, 3924)),
    preflight_seeds=[3800, 3801],
    laws=list(LAWS),
    n=64,
    loci=64,
    traits_per_view=4,
    n_pcs=2,
    max_k=4,
    min_cluster_size=5,
    n_init=10,
    model_seed=301,
    stability_draws=3,
    cap_bytes=1024**3,
    free_reserve=8 * 1024**3,
    serial_seconds=1800,
    allowance=1.25,
)


def density_audit(directory, arm):
    s = read_json(directory / "summary.json")
    with np.load(directory / "mixture.npz", allow_pickle=False) as a:
        w, mu = a["weights"], a["means"]
        if arm.family == "genomic":
            scores = np.load(directory / "pca_scores.npy", allow_pickle=False)
            membership = np.load(directory / "membership.npy", allow_pickle=False)
            cov = np.asarray([np.diag(v) for v in a["variances"]])
        else:
            scores, membership = a["scores"], a["membership"]
            cov = a["covariances"]
    n, d = scores.shape
    k = len(w)
    terms = []
    for weight, mean, c in zip(w, mu, cov, strict=True):
        sign, logdet = np.linalg.slogdet(c)
        assert sign == 1 and np.linalg.eigvalsh(c).min() > 0
        residual = scores - mean
        terms.append(
            np.log(weight)
            - 0.5
            * (
                d * np.log(2 * np.pi)
                + logdet
                + np.einsum("ij,ji->i", residual, np.linalg.solve(c, residual.T))
            )
        )
    terms = np.asarray(terms).T
    total = logsumexp(terms, axis=1)
    np.testing.assert_allclose(np.exp(terms - total[:, None]), membership, rtol=1e-7, atol=1e-9)
    params = (
        k
        - 1
        + k * d
        + (
            k * d
            if arm.family == "genomic"
            else (d * (d + 1) // 2 if arm.model.covariance_type == "tied" else k * d * (d + 1) // 2)
            if arm.family == "integrative"
            else k * d * (d + 1) // 2
        )
    )
    bic = float(-2 * total.sum() + params * np.log(n))
    candidates = (
        s["candidates"] if arm.family == "genomic" else read_json(directory / "candidates.json")
    )
    selected = next(r for r in candidates if r["k"] == s["selected_k"])
    assert selected["status"] == "eligible"
    np.testing.assert_allclose(bic, selected["bic"], rtol=1e-8, atol=1e-7)
    assert selected["bic"] == min(r["bic"] for r in candidates if r["status"] == "eligible")
    return dict(passed=True, bic=bic, selected_k=s["selected_k"], n=n)


def arrays_equal(left, right):
    count = 0
    for name in (
        "labels.npy",
        "membership.npy",
        "pca_scores.npy",
        "preprocessing.npz",
        "mixture.npz",
    ):
        a, b = left / name, right / name
        if not a.exists():
            assert not b.exists()
            continue
        if name.endswith(".npy"):
            np.testing.assert_array_equal(
                np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
            )
            count += 1
        else:
            with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
                assert x.files == y.files
                for key in x.files:
                    np.testing.assert_array_equal(x[key], y[key])
                    count += 1
    return count


def study(root, phase):
    root = Path(root)
    root.mkdir(exist_ok=True)
    frozen = root / "settings.json"
    if frozen.exists():
        assert read_json(frozen) == SETTINGS
    else:
        write_json(SETTINGS, frozen)
    if phase == "main":
        assert read_json(root / "resource-gate.json")["passed"]
    out = root / phase
    out.mkdir()
    start = time.monotonic()
    records = []
    audits = []
    parity_arrays = 0
    seeds = SETTINGS["preflight_seeds"] if phase == "preflight" else SETTINGS["main_seeds"]
    for index, seed in enumerate(seeds):
        law = LAWS[index % 6] if phase == "main" else ("common", "complementary")[index]
        draws = 3 if phase == "preflight" or index < 6 else 0
        source = out / f"{index:02d}-{law}-source"
        d, c = invented(source, seed=seed, law=law, draws=draws)
        destination = out / f"{index:02d}-{law}-analysis"
        summary = run_genomic_analysis(
            d, analysis_config=c, base_dir=source, output_dir=destination
        )
        manifest = read_json(destination / "manifest.json")
        dataset = config_from_dict(manifest["frozen_dataset"])
        cohort = manifest["cohort"]
        verify_sources(destination / "inputs", manifest["sources"])
        truth = read_json(source / "truth-evaluation-only.json")
        labels = dict(zip(truth["unit_ids"], truth["labels"], strict=True))
        for arm, row in zip(c.arms, summary["models"], strict=True):
            record = dict(
                index=index,
                seed=seed,
                law=law,
                name=arm.name,
                status=row["status"],
                selected_k=row["selected_k"],
                error=row["error"],
                truth_ari=None,
            )
            if row["status"] != "failed":
                native = destination / "arms" / arm.name
                point = read_genomic_arm(arm, native, cohort, dataset)["point"]
                # Truth enters only after all native fitting, never model selection.
                record["truth_ari"] = float(
                    adjusted_rand_score([labels[u] for u in cohort["unit_ids"]], point["labels"])
                )
                point_arm = replace(arm, model=replace(arm.model, n_bootstrap=0))
                replay = out / f"{index:02d}-{law}-replay-{arm.name}"
                run_genomic_arm(point_arm, dataset, destination / "inputs", replay)
                check = read_genomic_arm(point_arm, replay, cohort, dataset)
                assert point == check["point"]
                parity_arrays += arrays_equal(native, replay)
                audits.append(density_audit(native, arm))
                assert row["stability"]["requested"] == draws
                if arm.family == "genomic":
                    assert (
                        row["stability"]["successful"] + len(row["stability"]["failures"]) == draws
                    )
                else:
                    assert row["stability"]["successful"] + row["stability"]["failed"] == draws
                verify_sources(destination / "inputs", manifest["sources"])
            records.append(record)
        # Independent library agreement check, including degenerate single clusters.
        for pair in summary["comparisons"]:
            if pair["status"] == "available":
                a = read_json(destination / "arms" / pair["left"] / "comparison-point.json")[
                    "labels"
                ]
                b = read_json(destination / "arms" / pair["right"] / "comparison-point.json")[
                    "labels"
                ]
                np.testing.assert_allclose(
                    pair["adjusted_rand"], adjusted_rand_score(a, b), atol=1e-14
                )
        assert summary["inference"] == dict(status="not_computed", p_value=None)
        print(
            json.dumps(
                dict(phase=phase, completed=index + 1, total=len(seeds), status=summary["status"])
            ),
            flush=True,
        )
    elapsed = time.monotonic() - start
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    result = dict(
        passed=True,
        phase=phase,
        observations=len(seeds),
        point_pipelines=10 * len(seeds),
        bootstrap_refits=30 if phase == "preflight" else 90,
        records=records,
        density_audits=audits,
        exact_parity_arrays=parity_arrays,
        elapsed_seconds=elapsed,
        retained_bytes=size,
        failed_points=sum(r["status"] == "failed" for r in records),
    )
    write_json(result, out / "summary.json")
    if phase == "preflight":
        # Main has twelve times the point count and three times the draw count.
        projection = 1.25 * 12 * elapsed
        byte_projection = int(1.25 * 12 * size + 32 * 1024**2)
        free = shutil.disk_usage(root).free
        gate = dict(
            passed=projection < 1800 and byte_projection < 1024**3 and free >= 8 * 1024**3,
            projected_serial_seconds=projection,
            projected_bytes=byte_projection,
            free_bytes=free,
        )
        write_json(gate, root / "resource-gate.json")
        assert gate["passed"], gate
    else:
        assert elapsed < 1800 and size < 1024**3 and shutil.disk_usage(root).free >= 8 * 1024**3
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("preflight", "main"))
    args = parser.parse_args()
    study(args.output_dir, args.phase)
