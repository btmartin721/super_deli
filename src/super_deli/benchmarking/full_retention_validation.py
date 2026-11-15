"""Retrospective retained-evidence audits without refitting scientific models."""

import time
from pathlib import Path

import numpy as np

from super_deli.benchmarking.full_retention_audit import audit_run, close
from super_deli.benchmarking.signal_loss_fit import audit_candidates
from super_deli.core.partition_comparison import compare_partitions
from super_deli.io.config import read_json


def _truth_diagnostic(case):
    case = Path(case)
    result = read_json(case / "truth_initialized/summary.json")
    if result["status"] == "unavailable":
        assert not (case / "retained_all/summary.json").exists()
        return 0
    count = audit_candidates(case / "truth_initialized/candidates")
    summary = read_json(case / "retained_all/summary.json")
    truth = read_json(case.with_name(case.name + "-inputs") / "truth-evaluation-only.json")
    lookup = dict(zip(truth["unit_ids"], truth["labels"], strict=True))
    y = np.asarray([lookup[u] for u in summary["unit_ids"]])
    with np.load(case / "retained_all/mixture.npz", allow_pickle=False) as f:
        x = f["scores"].copy()
    with np.load(case / "truth_initialized/initialization.npz", allow_pickle=False) as f:
        np.testing.assert_array_equal(f["truth"], y)
        means = np.stack([x[y == j].mean(0) for j in range(4)])
        close(f["means"], means)
        close(f["weights"], np.bincount(y, minlength=4) / len(y))
        residual = x - means[y]
        expected = np.cov(residual, rowvar=False, bias=True)
        expected += summary["provenance"]["configuration"]["model"]["reg_covar"] * np.eye(
            x.shape[1]
        )
        close(expected, f["covariance"])
    assert result["selected"] is False and result["truth_informed"] is True
    fits = read_json(case / "truth_initialized/candidates/index.json")
    assert len(fits) == 1 and fits[0]["n_init"] == 1 and fits[0]["k"] == 4
    assert fits[0]["seed"] == int(
        np.random.SeedSequence([summary["point_seed"], 4]).generate_state(1)[0]
    )
    if result["status"] == "complete":
        fit = fits[0]
        assert fit["fit_status"] == "complete" and fit["converged"]
        close(fit["bic"], result["bic"])
        close(fit["log_likelihood"], result["log_likelihood"])
        with np.load(
            case / "truth_initialized/candidates" / fit["arrays"], allow_pickle=False
        ) as f:
            sizes = np.bincount(f["labels"], minlength=4)
            np.testing.assert_array_equal(sizes, result["sizes"])
            assert result["eligible"] == bool(
                min(sizes) >= summary["provenance"]["configuration"]["model"]["min_cluster_size"]
            )
    return count


def audit_phase(root, records, diagnostics):
    start = time.monotonic()
    root = Path(root)
    audits = []
    for path in sorted(root.rglob("validation.json")):
        row = read_json(path)
        audited = audit_run(path.parent)
        assert audited == row["audit"]
        audits.append(audited)
    for record in records:
        if record["origin"] != "fresh":
            continue
        task = record["task"]
        case = root / f"fresh-{task['index']:03d}"
        truth = read_json(case.with_name(case.name + "-inputs") / "truth-evaluation-only.json")
        for name, row in record["arms"].items():
            if row["status"] != "complete":
                continue
            summary = read_json(case / name / "summary.json")
            with np.load(case / name / "mixture.npz", allow_pickle=False) as f:
                lookup = dict(zip(truth["unit_ids"], truth["labels"], strict=True))
                y = np.asarray([lookup[u] for u in summary["unit_ids"]])
                predicted = f["labels"]
                if task["law"] == "continuous_control":
                    assert row["target_k"] is None and row["ari"] is None
                    assert row["correct_recovery"] is None
                    assert row["control_split"] == bool(len(set(predicted)) > 1)
                else:
                    target = y // 2 if task["law"] == "genomic_only" else y
                    expected = compare_partitions(
                        target.tolist(), predicted.tolist(), [True] * len(y), [True] * len(y)
                    )["adjusted_rand"]
                    close(row["ari"], expected)
                    assert row["target_k"] == len(set(target))
                    assert row["correct_recovery"] == (
                        len(set(predicted)) == len(set(target)) and expected >= 0.8
                    )
        # Identical paired draws are required across all completed arms.
        complete = [n for n, r in record["arms"].items() if r["status"] == "complete"]
        if task["stability"] and len(complete) > 1:
            paired = [read_json(case / n / "bootstrap_replicates.json") for n in complete]
            for reps in paired[1:]:
                assert [r["training_indices"] for r in reps] == [
                    r["training_indices"] for r in paired[0]
                ]
    for d in diagnostics:
        case = Path(d["path"])
        if d["baseline"]["status"] == d["hundred_starts"]["status"] == "complete":
            with (
                np.load(case / "retained_all/mixture.npz", allow_pickle=False) as a,
                np.load(case / "hundred_starts/mixture.npz", allow_pickle=False) as b,
            ):
                np.testing.assert_array_equal(a["scores"], b["scores"])
            assert d["hundred_starts"]["configuration"]["n_init"] == 100
            assert d["baseline"]["configuration"]["n_init"] == 10
    truth_checks = sum(_truth_diagnostic(d["path"]) for d in diagnostics)
    return dict(
        passed=True,
        runs=len(audits),
        component_checks=sum(a["component_checks"] for a in audits),
        observed_withheld=sum(a["point_withheld"] for a in audits),
        bootstrap_draws=sum(a.get("draws", 0) for a in audits),
        failed_draws=sum(a.get("failed_draws", 0) for a in audits),
        truth_component_checks=truth_checks,
        retained_space_checks=sum(a.get("retained_space_checks", 0) for a in audits),
        shape_refit_checks=sum(a.get("shape_complete_checks", 0) for a in audits),
        density_checks=sum(a.get("density_checks", 0) for a in audits),
        failed_component_attempts=sum(a.get("failed_component_attempts", 0) for a in audits),
        attempted_pipeline_refits=sum(max(1, a["events"]) for a in audits),
        completed_pipeline_refits=sum(
            0 if a["point_withheld"] else 1 + a.get("draws", 0) - a.get("failed_draws", 0)
            for a in audits
        ),
        seconds=time.monotonic() - start,
    )
