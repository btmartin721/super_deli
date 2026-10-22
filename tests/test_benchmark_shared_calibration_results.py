"""Portable evidence, tail calculations and scope of shared-family calibration."""

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from super_deli.models.joint_config import JointStudyConfig

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/shared-calibration/results"


def read(name):
    return json.loads((RESULTS / name).read_text())


def rows(name):
    with (RESULTS / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_prespecified_design_and_fit_accounting():
    summary, study = read("summary.json"), read("study.json")
    calibration = study["calibration"]
    assert JointStudyConfig.from_dict(summary["study"]) == JointStudyConfig.from_dict(study)
    assert study["random_seed"] == 20261031 and study["n_units"] == 64
    assert summary["datasets"] == 125 and study["n_datasets"] == 25
    assert calibration["n_simulations"] == 99 and calibration["alpha"] == 0.05
    model = calibration["model"]
    assert model["covariance_type"] == "tied" and model["random_seed"] == 71323
    assert model["n_init"] == 10 and model["n_pcs"] == 2 and model["max_k"] == 4
    assert model["min_cluster_size"] == 5 and model["n_bootstrap"] == 0
    assert len(calibration["block_ids"]) == 128 and len(set(calibration["block_ids"])) == 16
    assert len(calibration["predictor_blocks"]) == 4
    assert summary["planned_fits"] == 12500
    assert (
        sum(summary[key] for key in ("successful_fits", "failed_fits", "unattempted_fits")) == 12500
    )
    execution = read("execution.json")
    assert execution["workers"] == 4 and execution["threads_per_job"] == 1
    assert execution["replacement_datasets"] == execution["replacement_draws"] == 0


def test_portable_hashes_and_immutable_frozen_runtime():
    manifest = read("manifest.json")
    assert manifest["local_artifact_id"] == "shared-null-20261022"
    for name, expected in manifest["published_artifacts"].items():
        assert Path(name).name == name
        assert hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == expected
    for name, expected in manifest["publication_files"].items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    for path in RESULTS.iterdir():
        assert "/Users/" not in path.read_text() and "/private/tmp" not in path.read_text()
    sources = read("runtime_sources.json")
    assert len(sources) == 115
    assert all(not Path(p).is_absolute() and len(h) == 64 for p, h in sources.items())
    versions = read("runtime_versions.json")
    assert versions["libraries"]["super-deli"]["version"] == "0.24.0.dev0"
    assert all(set(value) == {"version"} for value in versions["libraries"].values())


def test_independent_plus_one_tails_and_conditional_binomial_intervals():
    diagnostics = rows("diagnostics.csv")
    assert len(diagnostics) == 125
    for row in diagnostics:
        if row["status"] != "available":
            assert row["p_value"] == ""
            continue
        n, x = int(row["successful_draws"]), int(row["exceedances"])
        assert n == 99 and int(row["failed_draws"]) == 0
        p = float(row["p_value"])
        assert p == (1 + x) / (1 + n)
        assert (row["reject"] == "True") == (p <= 0.05)
        low, high = float(row["tail_lower_95"]), float(row["tail_upper_95"])
        if x:
            upper_tail = sum(
                math.comb(n, j) * low**j * (1 - low) ** (n - j) for j in range(x, n + 1)
            )
            assert upper_tail == pytest.approx(0.025, abs=1e-9)
        else:
            assert low == 0
        if x < n:
            lower_tail = sum(
                math.comb(n, j) * high**j * (1 - high) ** (n - j) for j in range(x + 1)
            )
            assert lower_tail == pytest.approx(0.025, abs=1e-9)
        else:
            assert high == 1
        assert int(row["design_rank"]) == 5 and int(row["residual_df"]) == 59
        assert float(row["residual_correlation_min_eigenvalue"]) > 0


def test_independent_rejection_rates_sizes_and_interpretation():
    data, summary = rows("datasets.csv"), read("summary.json")
    assert len(data) == 125
    for rate in summary["rates"]:
        selected = [row for row in data if row["case"] == rate["case"]]
        assert len(selected) == 25
        assert {int(row["repetition"]) for row in selected} == set(range(25))
        available = [row for row in selected if row["p_value"]]
        events = sum(float(row["p_value"]) <= 0.05 for row in available)
        assert rate["n_available"] == len(available) and rate["events"] == events
        n, z = len(available), 1.959963984540054
        if n:
            p = events / n
            center = (p + z * z / (2 * n)) / (1 + z * z / n)
            radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            assert rate["wilson_95"] == pytest.approx(
                [max(0, center - radius), min(1, center + radius)]
            )
        for k, count in summary["selected_k_counts"][rate["case"]].items():
            assert count == sum(row["selected_k"] == k for row in selected)
    report = (ROOT / "docs/shared-covariance-calibration-results.md").read_text().lower()
    for phrase in (
        "approximate",
        "no general",
        "species",
        "not independently refitted",
        "12,500",
        "stability",
        "fresh",
        "unpaired",
    ):
        assert phrase in report
