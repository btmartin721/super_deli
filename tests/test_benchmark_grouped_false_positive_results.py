"""Portable record integrity and independent, dependency-free statistical checks."""

import hashlib
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/grouped-false-positive/results"


def read(name):
    return json.loads((DATA / name).read_text())


def test_prespecified_design_and_separate_seed_domains():
    study, design, seeds = (
        read("study.json"),
        read("design-summary.json"),
        read("seed-domain-check.json"),
    )
    assert study["n_datasets"] == design["datasets_per_case"] == 500
    assert design["new_unique_pipeline_fits"] == 497500
    assert study["n_units"] == 64 and study["calibration"]["n_simulations"] == 99
    model = study["calibration"]["model"]
    assert model["covariance_type"] == "full" and model["selection_criterion"] == "aic"
    assert model["n_pcs"] == 4 and model["max_k"] == 4 and model["n_init"] == 10
    assert model["random_seed"] == 71323 and model["n_bootstrap"] == 0
    assert model["views"][0]["pca_fit"] == "target_means"
    assert design["seed_records_retained_locally"] == 2500
    assert seeds["fresh_data_seeds"] == seeds["fresh_null_streams"] == 2500
    assert seeds["all_distinct"] and seeds["disjoint_from_retained_prior_seeds"]
    assert seeds["fresh_paired_draw_seeds"] == 247500


def test_portable_checksums_and_frozen_runtime_history():
    manifest = read("manifest.json")
    assert set(manifest) == {p.name for p in DATA.iterdir() if p.is_file()} - {"manifest.json"}
    for name, expected in manifest.items():
        value = (DATA / name).read_bytes()
        assert len(value) == expected["size_bytes"], name
        assert hashlib.sha256(value).hexdigest() == expected["sha256"], name
    before, after = read("runtime_sources.json"), read("audit_sources.json")
    assert len(before) == 138 and len(after) == 139
    assert {p for p in before if before[p] != after[p]} == {"__init__.py", "cli.py"}
    assert set(after) - set(before) == {"benchmarking/grouped_false_positive.py"}
    # These compare frozen historical manifests, not future package source bytes.


def binomial_cdf(k, n, probability):
    return sum(
        math.comb(n, i) * probability**i * (1 - probability) ** (n - i) for i in range(k + 1)
    )


def test_exact_primary_intervals_by_independent_binomial_equations():
    summary = read("summary.json")
    assert len(summary["primary_rates"]) == 10
    for row in summary["primary_rates"]:
        events, n = row["events"], row["available"]
        assert n == row["requested"] == 500 and row["withheld"] == 0
        assert row["rate_among_available"] == events / n
        for key, tail in (("exact_interval_95", 0.025), ("bonferroni_familywise_95", 0.0025)):
            lower, upper = row[key]
            assert 0 <= lower <= events / n <= upper <= 1
            assert 1 - binomial_cdf(events - 1, n, lower) == pytest.approx(tail, abs=1e-9)
            assert binomial_cdf(events, n, upper) == pytest.approx(tail, abs=1e-9)
        assert row["bonferroni_familywise_95"][0] <= row["exact_interval_95"][0]
        assert row["bonferroni_familywise_95"][1] >= row["exact_interval_95"][1]


def test_detailed_evidence_and_private_metadata_are_excluded():
    names = {p.name for p in DATA.iterdir()}
    assert not any(name.startswith("pipelines-") for name in names)
    assert not names & {
        "tails.csv",
        "covariance.csv",
        "observations.csv",
        "prespecified-design.json",
        "evidence-index.json",
    }
    assert "full_design_sha256" not in read("design-summary.json")
    assert "cleanup_archive_sha256" not in read("preservation-summary.json")
    assert sum(p.stat().st_size for p in DATA.iterdir()) < 100000
    assert read("independent-tails.json")["tails_checked"] == 10000


def test_preservation_and_unique_successful_fit_accounting():
    summary, preservation = read("summary.json"), read("preservation-summary.json")
    assert summary["observations"] == 2500
    assert summary["fit_accounting"] == {
        "attempted_fits": 497500,
        "failed_fits": 0,
        "planned_fits": 497500,
        "successful_fits": 497500,
        "unattempted_fits": 0,
    }
    assert summary["archive_members"] == 15985000
    assert read("independent-tails.json")["tails_checked"] == 10000
    assert preservation["before_after_fingerprints_identical"]
    assert preservation["prior_study_files_verified"] == 216962
    assert preservation["prior_checkouts_verified"] == 7
    assert preservation["original_tps_sources_present_at_start"] == 0
    assert (
        sum(
            p["known_only_rejections"] + p["fitted_only_rejections"]
            for p in summary["paired_decisions"]
        )
        == 58
    )
