"""Integrity and interpretation of the frozen public joint-calibration evidence."""

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/joint-calibration/results"


def read(name):
    return json.loads((RESULTS / name).read_text())


def test_study_design_and_full_accounting():
    summary = read("summary.json")
    c = summary["study"]["calibration"]
    assert summary["datasets"] == 125
    assert summary["planned_fits"] == summary["successful_fits"] == 12500
    assert summary["failed_fits"] == summary["unattempted_fits"] == 0
    assert c["n_simulations"] == 99 and c["alpha"] == 0.05
    assert c["model"]["n_init"] == 10 and c["model"]["max_k"] == 4
    assert c["model"]["n_bootstrap"] == 0
    assert len(c["block_ids"]) == 128 and len(set(c["block_ids"])) == 16
    assert len(c["predictor_blocks"]) == 4


def test_public_hashes_and_private_path_exclusion():
    manifest = read("manifest.json")
    assert "local_artifact_root" not in manifest
    assert manifest["local_artifact_id"] == "joint-null-20261021"
    for name, digest in manifest["published_artifacts"].items():
        assert Path(name).name == name
        assert hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == digest
    for name in ("manifest.json", "runtime_versions.json", "execution.json"):
        assert "/Users/" not in (RESULTS / name).read_text()
    for name, digest in manifest["publication_files"].items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    versions = read("runtime_versions.json")
    assert all(set(v) == {"version"} for v in versions["libraries"].values())
    sources = read("runtime_sources.json")
    assert len(sources) == 106
    for relative, digest in sources.items():
        assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
        assert hashlib.sha256((ROOT / "src" / relative).read_bytes()).hexdigest() == digest


def test_independent_rate_reconstruction_and_wilson_intervals():
    with (RESULTS / "datasets.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 125
    for rate in read("summary.json")["rates"]:
        selected = [r for r in rows if r["case"] == rate["case"]]
        assert len(selected) == 25 and {int(r["repetition"]) for r in selected} == set(range(25))
        assert all(r["status"] == "available" for r in selected)
        events = sum(float(r["p_value"]) <= 0.05 for r in selected)
        assert events == rate["events"] and rate["n_available"] == 25 and rate["n_missing"] == 0
        p = events / 25
        z = 1.959963984540054
        center = (p + z * z / 50) / (1 + z * z / 25)
        radius = z * math.sqrt(p * (1 - p) / 25 + z * z / (4 * 25 * 25)) / (1 + z * z / 25)
        assert rate["wilson_95"] == pytest.approx(
            [max(0, center - radius), min(1, center + radius)]
        )
        for row in selected:
            assert 0.01 <= float(row["p_value"]) <= 1
            assert (row["reject"] == "True") == (float(row["p_value"]) <= 0.05)


def test_diagnostic_dimensions_and_scope():
    with (RESULTS / "diagnostics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 125
    assert all(int(r["design_rank"]) == 5 and int(r["residual_df"]) == 59 for r in rows)
    assert all(float(r["residual_correlation_min_eigenvalue"]) > 0 for r in rows)
    report = (ROOT / "docs/joint-genomic-phenotypic-calibration-results.md").read_text()
    assert "no general" in report.lower() or "do not establish general" in report.lower()
    assert "not an association" in report
    assert "not independently refitted" in report
    assert "125/125" in report and "12,500/12,500" in report
    assert "never selected K=4" in report
