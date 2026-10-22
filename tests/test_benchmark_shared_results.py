"""Portable evidence checks for the completed covariance-family comparison."""

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/shared-covariance/results"


def read(name):
    return json.loads((RESULTS / name).read_text())


def rows(name):
    with (RESULTS / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_artifact_hashes_and_private_data_exclusion():
    manifest = read("manifest.json")
    assert manifest["local_artifact_id"] == "shared-covariance-20261022"
    assert manifest["obsolete_tracked_files_removed"] == 83
    for name, digest in manifest["published_artifacts"].items():
        assert Path(name).name == name
        assert hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["publication_files"].items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    assert all("/Users/" not in p.read_text() for p in RESULTS.iterdir())
    sources = read("runtime_sources.json")
    assert len(sources) == 115
    assert all(
        not Path(name).is_absolute() and len(digest) == 64 for name, digest in sources.items()
    )
    assert read("runtime_versions.json")["libraries"]["super-deli"]["version"] == "0.23.0.dev0"
    assert not any(
        (ROOT / name).exists()
        for name in ["impute", "mem_profiler", "training_data", "super_deli_datasets"]
    )


def test_frozen_design_and_complete_denominators():
    result = read("summary.json")
    study = read("study.json")
    assert result["datasets"] == 150
    assert result["planned_points"] == result["successful_points"] == 300
    assert result["planned_bootstrap"] == result["successful_bootstrap"] == 500
    assert (
        result["failed_points"]
        == result["failed_bootstrap"]
        == result["unattempted_bootstrap"]
        == result["significance_fits"]
        == 0
    )
    assert result["reproduced_archived"] == 25 and result["verified_files"] == 15352
    assert (
        study["model"]["n_pcs"] == 2
        and study["model"]["n_init"] == 10
        and study["model"]["random_seed"] == 71323
    )
    assert "covariance_type" not in study["model"]
    datasets = read("datasets.json")
    assert len(datasets) == len({r["seed"] for r in datasets}) == 150
    stability = rows("stability.csv")
    assert len(stability) == 20 and sum(int(r["successful"]) for r in stability) == 500


def test_recovery_rates_intervals_and_paired_direction():
    data = rows("points.csv")
    summary = read("summary.json")
    assert len(data) == 300
    for entry in summary["summaries"]:
        selected = [
            r for r in data if (r["cohort"], r["family"]) == (entry["cohort"], entry["family"])
        ]
        assert len(selected) == entry["available"] == 25
        counts = {str(k): sum(int(r["selected_k"]) == k for r in selected) for k in range(1, 5)}
        assert entry["selected_k_counts"] == counts
        aris = sorted(float(r["ari"]) for r in selected if r["ari"])
        assert entry["median_ari"] == (aris[12] if aris else None)
        assert (
            entry["mean_ari"] == pytest.approx(sum(aris) / 25)
            if aris
            else entry["mean_ari"] is None
        )
        z, p, n = 1.959963984540054, counts["4"] / 25, 25
        center = (p + z * z / (2 * n)) / (1 + z * z / n)
        radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        assert entry["selected_four_wilson_95"] == pytest.approx(
            [max(0, center - radius), min(1, center + radius)]
        )
    lookup = {(r["cohort"], r["family"]): r for r in summary["summaries"]}
    assert lookup["fresh_64", "full"]["selected_k_counts"]["4"] == 0
    assert lookup["fresh_64", "tied"]["selected_k_counts"]["4"] == 8
    assert lookup["fresh_256", "full"]["selected_k_counts"]["4"] == 16
    assert lookup["fresh_256", "tied"]["selected_k_counts"]["4"] == 25
    assert (
        lookup["unequal_variance", "tied"]["mean_ari"]
        < lookup["unequal_variance", "full"]["mean_ari"]
    )
    assert lookup["cline", "tied"]["selected_k_counts"]["1"] == 1


def test_family_specific_bic_and_report_scope():
    candidates = rows("candidates.csv")
    assert len(candidates) == 1200
    for row in candidates:
        k, d, n = (int(row[name]) for name in ("k", "d", "n"))
        p = k * d + k - 1 + d * (d + 1) // 2 * (1 if row["family"] == "tied" else k)
        assert int(row["parameters"]) == p
        assert float(row["penalty"]) == pytest.approx(p * math.log(n))
        assert float(row["bic"]) == pytest.approx(
            -2 * float(row["log_likelihood_from_bic"]) + p * math.log(n)
        )
    report = (ROOT / "docs/shared-covariance-clustering-results.md").read_text().lower()
    for phrase in [
        "300/300",
        "500/500",
        "not independently refitted",
        "partial improvement",
        "unequal-variance",
        "default",
        "p-values",
    ]:
        assert phrase in report
