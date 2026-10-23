"""Portable full-family group-mean evidence checks settings, tails and denominators."""

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from super_deli.benchmarking.config import derived_seed
from super_deli.models.grouped_config import GroupedStudyConfig, study_cases

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/full-groupmean-calibration/results"


def read(name):
    return json.loads((RESULTS / name).read_text())


def rows(name):
    with (RESULTS / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_frozen_settings_seeds_and_pipeline_accounting():
    c = GroupedStudyConfig.from_dict(read("study.json"))
    result = read("summary.json")
    assert c == GroupedStudyConfig.from_dict(result["study"])
    assert c.random_seed == 20261110 and (c.n_units, c.n_datasets) == (64, 25)
    assert len(study_cases(c)) == 7 and c.calibration.n_simulations == 99
    assert c.calibration.schema_version == 2 and c.calibration.alpha == 0.05
    m = c.calibration.model
    assert (m.n_pcs, m.max_k, m.min_cluster_size, m.n_init, m.n_bootstrap) == (4, 4, 5, 10, 0)
    assert m.covariance_type == "full" and m.random_seed == 71323
    assert all(v.n_pcs == 2 and v.weight == 1 for v in m.views)
    assert [v.pca_fit for v in m.views] == ["target_means", "source_rows"]
    assert c.calibration.solver.estimator == "conditional_psd_v1"
    data = rows("datasets.csv")
    assert len(data) == result["datasets"] == 175 and result["planned_fits"] == 17500
    assert len({r["data_seed"] for r in data}) == 175
    for name in ("shared-grouped-calibration", "groupmean-calibration"):
        prior = ROOT / f"examples/{name}/results/datasets.csv"
        with prior.open(newline="") as handle:
            assert not (
                {r["data_seed"] for r in data} & {r["data_seed"] for r in csv.DictReader(handle)}
            )
    for r in data:
        assert int(r["data_seed"]) == derived_seed(
            c.random_seed, r["case"], int(r["repetition"]), "observed_data"
        )
    for key in (
        "planned_fits",
        "attempted_fits",
        "successful_fits",
        "failed_fits",
        "unattempted_fits",
    ):
        assert result[key] == sum(int(r[key]) for r in data)
    assert result["successful_fits"] + result["failed_fits"] + result["unattempted_fits"] == 17500
    e = read("execution.json")
    assert (e["workers"], e["threads_per_job"]) == (4, 1)
    assert e["replacement_datasets"] == e["replacement_draws"] == 0


def test_integrity_packaging_and_no_private_paths():
    manifest = read("manifest.json")
    assert manifest["local_artifact_id"] == "full-groupmean-null-20261023"
    for n, h in manifest["published_artifacts"].items():
        assert Path(n).name == n and hashlib.sha256((RESULTS / n).read_bytes()).hexdigest() == h
    for n, h in manifest["publication_files"].items():
        assert not Path(n).is_absolute() and ".." not in Path(n).parts
        assert hashlib.sha256((ROOT / n).read_bytes()).hexdigest() == h
    for p in RESULTS.iterdir():
        assert "/Users/" not in p.read_text() and "/private/tmp" not in p.read_text()
    assert len(read("runtime_sources.json")) == 124
    assert read("runtime_versions.json")["libraries"]["super-deli"]["version"] == "0.29.0.dev0"


def test_independent_plus_one_tails_and_conditional_interval_equations():
    for r in rows("diagnostics.csv"):
        if r["status"] != "available":
            assert r["p_value"] == r["reject"] == ""
            continue
        n, x = int(r["successful_draws"]), int(r["exceedances"])
        assert n == 99 and int(r["failed_draws"]) == 0
        p = (1 + x) / (1 + n)
        assert float(r["p_value"]) == float(r["failed_p_lower"]) == float(r["failed_p_upper"]) == p
        assert (r["reject"] == "True") == (p <= 0.05)
        low, high = float(r["tail_lower_95"]), float(r["tail_upper_95"])
        if x:
            assert sum(
                math.comb(n, j) * low**j * (1 - low) ** (n - j) for j in range(x, n + 1)
            ) == pytest.approx(0.025, abs=1e-9)
        else:
            assert low == 0
        if x < n:
            assert sum(
                math.comb(n, j) * high**j * (1 - high) ** (n - j) for j in range(x + 1)
            ) == pytest.approx(0.025, abs=1e-9)
        else:
            assert high == 1


def test_independent_rates_counts_rank_acceptance_and_scope():
    result, data, diagnostics = read("summary.json"), rows("datasets.csv"), rows("diagnostics.csv")
    for r in diagnostics:
        if r["null_fit_available"] == "True":
            counts = [int(r[f"groups_with_{i}_specimens"]) for i in range(1, 5)]
            assert sum(counts) == int(r["n_groups"]) == int(r["n_ecology_rows"]) == 64
            assert sum((i + 1) * n for i, n in enumerate(counts)) == int(r["n_specimens"])
            assert 0 <= int(r["conditional_rank"]) <= 4 and float(r["optimality_maximum"]) <= 1e-5
    for rate in result["rates"]:
        selected = [r for r in data if r["case"] == rate["case"]]
        available = [r for r in selected if r["p_value"]]
        n, x = len(available), sum(float(r["p_value"]) <= 0.05 for r in available)
        assert len(selected) == 25 and (rate["n_available"], rate["events"]) == (n, x)
        assert rate["withheld_outcome_bounds"] == [x / 25, (x + 25 - n) / 25]
        if n:
            z = 1.959963984540054
            p = x / n
            center = (p + z * z / (2 * n)) / (1 + z * z / n)
            radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            assert rate["wilson_95"] == pytest.approx(
                [max(0, center - radius), min(1, center + radius)]
            )
        expected = {str(k): sum(r["selected_k"] == str(k) for r in selected) for k in range(1, 5)}
        assert result["selected_k_counts"][rate["case"]] == expected
    recovery = read("recovery_summary.json")
    values = sorted(float(r["ari"]) for r in rows("recovery.csv") if r["ari"])
    assert recovery["available"] == len(values) and recovery["requested"] == 25
    if values:
        mean = sum(values) / len(values)
        assert recovery["mean_ari"] == pytest.approx(mean)
        assert recovery["median_ari"] == pytest.approx(values[len(values) // 2])
        mcse = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1) / len(values))
        assert recovery["mean_mcse"] == pytest.approx(mcse)
    text = (ROOT / "docs/full-groupmean-calibration-results.md").read_text().lower()
    for phrase in (
        "nonregular",
        "approximate",
        "no general",
        "unpaired",
        "stability",
        "17,500",
        "counts",
    ):
        assert phrase in text
