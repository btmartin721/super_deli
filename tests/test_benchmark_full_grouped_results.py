"""Dependency-free checks of published diagnostic evidence and inference boundaries."""

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import pytest

from super_deli.benchmarking.config import derived_seed
from super_deli.models.full_grouped_diagnostic_config import FullGroupedDiagnosticConfig

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/full-grouped-diagnostics/results"


def read(name):
    return json.loads((RESULTS / name).read_text())


def rows(name):
    with (RESULTS / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_published_integrity_and_no_private_paths():
    manifest = read("manifest.json")
    assert manifest["local_artifact_id"] == "full-grouped-diagnostics-20261023"
    for name, digest in manifest["published_artifacts"].items():
        assert Path(name).name == name
        assert hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["figure_files"].items():
        assert (
            hashlib.sha256(
                (ROOT / "docs/figures/full-grouped-diagnostics" / name).read_bytes()
            ).hexdigest()
            == digest
        )
    assert (
        hashlib.sha256(
            (ROOT / "scripts/report_full_grouped_diagnostics.py").read_bytes()
        ).hexdigest()
        == manifest["report_script_sha256"]
    )
    for path in RESULTS.iterdir():
        assert "/Users/" not in path.read_text() and "/private/tmp" not in path.read_text()
    assert len(read("runtime_sources.json")) == 129
    assert read("runtime_versions.json")["libraries"]["super-deli"]["version"] == "0.30.0.dev0"


def test_frozen_seeds_settings_and_complete_fit_accounting():
    config = FullGroupedDiagnosticConfig.from_dict(read("study.json"))
    assert (config.random_seed, config.n_datasets, config.small_n, config.large_n) == (
        20261112,
        25,
        64,
        256,
    )
    source = read("source_study.json")
    data = read("datasets.json")
    assert len(data) == 75 and len({r["seed"] for r in data}) == 75
    for r in data:
        assert r["seed"] == derived_seed(
            source["random_seed"] if r["cohort"] == "archived" else config.random_seed,
            "complementary" if r["cohort"] == "archived" else r["cohort"],
            r["repetition"],
            "observed_data",
        )
    summary = read("summary.json")
    assert summary["successful_points"] == summary["planned_points"] == 450
    assert summary["successful_truth_initialized"] == summary["planned_truth_initialized"] == 225
    assert summary["archived_tests"] == 175 and summary["archived_draws"] == 17325
    assert (
        summary["failed_points"]
        == summary["failed_truth_initialized"]
        == summary["unattempted_truth_initialized"]
        == summary["bootstrap_or_calibration_fits"]
        == 0
    )
    assert summary["replayed_archived"] == 25 and summary["verified_files"] == 15689
    execution = read("execution.json")
    assert execution["status"] == "complete" and execution["workers"] == 4
    assert execution["threads_per_job"] == 1
    assert (
        execution["replacement_fits"]
        == execution["new_significance_draws"]
        == execution["new_stability_draws"]
        == 0
    )


def test_recovery_rates_and_mean_mcse_from_rows():
    data = rows("points.csv")
    assert len(data) == 450 and all(r["status"] == "complete" for r in data)
    for row in read("summary.json")["summaries"]:
        selected = [r for r in data if (r["cohort"], r["arm"]) == (row["cohort"], row["arm"])]
        assert len(selected) == row["requested"] == row["available"] == 25
        counts = {str(k): sum(int(r["selected_k"]) == k for r in selected) for k in range(1, 5)}
        assert counts == row["selected_k_counts"]
        ari = [float(r["ari"]) for r in selected]
        assert row["ari"]["mean"] == pytest.approx(statistics.mean(ari), abs=1e-12)
        assert row["ari"]["mean_mcse"] == pytest.approx(statistics.stdev(ari) / 5, abs=1e-12)
        p, n, z = counts["4"] / 25, 25, 1.959963984540054
        center = (p + z * z / (2 * n)) / (1 + z * z / n)
        radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        assert row["selected_four_wilson_95"] == pytest.approx(
            [max(0, center - radius), min(1, center + radius)], abs=1e-12
        )


def test_candidate_dimension_family_counts_and_oracle_separation():
    for r in rows("candidates.csv"):
        k, d, n = int(r["k"]), int(r["d"]), int(r["n"])
        q = k - 1 + k * d + (1 if r["arm"] == "shared_4" else k) * d * (d + 1) // 2
        assert int(r["parameters"]) == q
        assert float(r["penalty"]) == pytest.approx(q * math.log(n), abs=1e-10)
        if r["bic"]:
            assert float(r["bic_minus_selected"]) == pytest.approx(
                float(r["penalty_minus_selected"]) - float(r["twice_log_likelihood_gain"]), abs=1e-9
            )
    oracle = rows("truth_initialized.csv")
    assert len(oracle) == 225
    assert all(int(r["k"]) == (4 if r["partition"] == "four" else 2) for r in oracle)
    assert all(r["status"] == "complete" and r["eligible"] == "True" for r in oracle)
    four = [r for r in oracle if r["partition"] == "four"]
    assert len(four) == 75 and all(float(r["bic_minus_selected"]) > 0 for r in four)
    assert all(
        r["tested"] == r["exactly_identical"] == 25
        for r in read("mechanisms.json")["paired_scores"]
    )


def test_historical_tails_exactly_preserve_existing_inference():
    with (ROOT / "examples/full-groupmean-calibration/results/diagnostics.csv").open(
        newline=""
    ) as handle:
        prior = {(r["case"], r["repetition"]): r for r in csv.DictReader(handle)}
    data = rows("retained_tails.csv")
    assert len(data) == 175 and len(prior) == 175
    for r in data:
        old = prior[r["case"], r["repetition"]]
        assert r["p_value"] == old["p_value"] and r["reject"] == old["reject"]
        assert int(r["successful_draws"]) == 99 and int(r["failed_draws"]) == 0
        assert float(r["p_value"]) == (1 + int(r["exceedances"])) / 100
        if float(r["observed_statistic"]) == 0:
            assert float(r["p_value"]) == 1
    alternative = [r for r in data if r["case"] == "complementary"]
    assert sum(float(r["observed_statistic"]) == 0 for r in alternative) == 19
    assert sum(r["reject"] == "True" for r in alternative) == 6
    assert sum(int(r["null_zero_count"]) for r in alternative) == 2473
