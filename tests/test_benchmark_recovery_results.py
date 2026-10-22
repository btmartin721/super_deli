"""Reconstruct the portable diagnostic evidence without private raw artifacts."""

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/recovery-diagnostics/results"


def read(name):
    return json.loads((RESULTS / name).read_text())


def rows(name):
    with (RESULTS / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_fingerprints_design_and_preserved_scientific_sources():
    manifest = read("manifest.json")
    assert manifest["base_pr"] == 49
    assert manifest["local_artifact_id"] == "integrative-recovery-20261021"
    for name, digest in manifest["published_artifacts"].items():
        assert Path(name).name == name
        assert hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["publication_files"].items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    assert all("/Users/" not in p.read_text() for p in RESULTS.iterdir())
    old = json.loads((ROOT / "examples/joint-calibration/results/runtime_sources.json").read_text())
    frozen = read("runtime_sources.json")
    assert len(frozen) == 111
    preserved = {
        k: v for k, v in old.items() if k not in {"super_deli/__init__.py", "super_deli/cli.py"}
    }
    assert len(preserved) == manifest["compatible_historical_scientific_modules"] == 104
    assert all(frozen[k] == v for k, v in preserved.items())
    grid = read("grid.json")
    assert (
        len(grid) == 7 and grid["baseline"]["n_init"] == 10 and grid["starts_100"]["n_init"] == 100
    )
    assert all(v["n_bootstrap"] == 0 and v["max_k"] == 4 for v in grid.values())
    datasets = read("datasets.json")
    assert len(datasets) == len({r["seed"] for r in datasets}) == 75
    assert {r["n"] for r in datasets if r["cohort"] == "fresh_256"} == {256}


def test_point_summaries_and_descriptive_intervals():
    summary, data = read("summary.json"), rows("points.csv")
    assert summary["planned_points"] == summary["successful_points"] == len(data) == 525
    assert summary["failed_points"] == summary["failed_truth_initialized"] == 0
    assert summary["bootstrap_or_calibration_fits"] == 0 and summary["replayed_archived"] == 25
    assert len({(r["cohort"], r["repetition"], r["variant"]) for r in data}) == 525
    for entry in summary["summaries"]:
        selected = [
            r for r in data if (r["cohort"], r["variant"]) == (entry["cohort"], entry["variant"])
        ]
        assert len(selected) == entry["available"] == 25 and entry["failed"] == 0
        assert {
            str(k): sum(int(r["selected_k"]) == k for r in selected) for k in range(1, 5)
        } == entry["selected_k_counts"]
        agreement = sorted(float(r["ari"]) for r in selected)
        assert entry["ari"]["median"] == agreement[12]
        assert entry["ari"]["mean"] == pytest.approx(sum(agreement) / 25)
        events = entry["selected_k_counts"]["4"]
        z, p, n = 1.959963984540054, events / 25, 25
        center = (p + z * z / (2 * n)) / (1 + z * z / n)
        radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        assert entry["selected_four_wilson_95"] == pytest.approx(
            [max(0, center - radius), min(1, center + radius)]
        )


def test_bic_parameter_cost_and_optimization_separation():
    candidates = rows("candidates.csv")
    assert len(candidates) == 2100
    for row in candidates:
        k, d, n = (int(row[key]) for key in ("k", "d", "n"))
        p = k * (d + d * (d + 1) // 2) + k - 1
        assert p == int(row["parameters"])
        assert float(row["penalty"]) == pytest.approx(p * math.log(n))
        assert float(row["bic"]) == pytest.approx(
            -2 * float(row["log_likelihood_from_bic"]) + p * math.log(n)
        )
        assert float(row["bic_minus_selected"]) == pytest.approx(
            float(row["penalty_minus_selected"]) - float(row["twice_log_likelihood_gain"]),
            abs=1e-10,
        )
    truth = rows("truth_initialized.csv")
    assert len(truth) == 75 and {r["status"] for r in truth} == {"complete"}
    assert {r["eligible"] for r in truth} == {"True"}
    assert all(
        float(r["bic_minus_baseline_selected"]) > 0 for r in truth if r["cohort"] != "fresh_256"
    )
    assert (
        sum(
            float(r["bic_minus_baseline_selected"]) < 0 for r in truth if r["cohort"] == "fresh_256"
        )
        == 9
    )


def test_supplemental_dimensions_and_report_boundaries():
    retention, loadings, paired = rows("retention.csv"), rows("loadings.csv"), rows("paired.csv")
    assert len(retention) == 375 and len(loadings) == 150 and len(paired) == 18
    for row in loadings:
        assert sum(
            float(row[k]) for k in ("genomic", "morphological", "ecological")
        ) == pytest.approx(1)
    for row in paired:
        assert (
            sum(int(row[k]) for k in ("ari_increased", "ari_decreased", "ari_unchanged"))
            == int(row["n_paired"])
            == 25
        )
    report = (ROOT / "docs/integrative-recovery-diagnostics-results.md").read_text()
    for phrase in (
        "525/525",
        "75/75",
        "no stability or significance",
        "not independently refitted",
        "no model, default",
        "49.9066",
        "21/25 (0.969)",
    ):
        assert phrase in report.lower()
