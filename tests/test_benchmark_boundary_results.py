"""Protect boundary-study evidence, paired inputs and independent-dataset counts."""

import csv
import hashlib
import json
from pathlib import Path

from super_deli import __version__
from super_deli.benchmarking.config import derived_seed
from super_deli.models.grouped_config import GroupedStudyConfig, study_cases

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/boundary-calibration/results"


def read(name):
    return json.loads((RESULTS / name).read_text())


def rows(name):
    with (RESULTS / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_public_hashes_and_frozen_reporting_package_sources():
    m = read("manifest.json")
    for path, digest in m["public_artifacts"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    if __version__ == m["reporting_version"]:
        for path, digest in m["reporting_code_sha256"].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    legacy = json.loads((ROOT / "examples/grouped-calibration/results/manifest.json").read_text())
    assert m["legacy_raw_files_inventory_sha256"] == legacy["raw_files_inventory_sha256"]


def test_dataset_seeds_and_all_attempt_denominators():
    study = GroupedStudyConfig.from_dict(read("study.json"))
    data = rows("datasets.csv")
    s = read("summary.json")
    assert len(data) == len(study_cases(study)) * study.n_datasets == 175
    assert len({(r["case"], r["repetition"]) for r in data}) == 175
    assert len({r["data_seed"] for r in data}) == len({r["null_stream_seed"] for r in data}) == 175
    assert {r["data_seed"] for r in data}.isdisjoint({r["null_stream_seed"] for r in data})
    assert (
        s["planned_fits"],
        s["attempted_fits"],
        s["successful_fits"],
        s["failed_fits"],
        s["unattempted_fits"],
    ) == (17500, 17500, 17500, 0, 0)
    for key in (
        "planned_fits",
        "attempted_fits",
        "successful_fits",
        "failed_fits",
        "unattempted_fits",
    ):
        assert s[key] == sum(int(r[key]) for r in data)
    for r in data:
        assert int(r["data_seed"]) == derived_seed(
            study.random_seed, r["case"], int(r["repetition"]), "observed_data"
        )
        assert int(r["null_stream_seed"]) == derived_seed(
            study.random_seed, r["case"], int(r["repetition"]), "null_stream"
        )


def test_availability_rates_and_covariance_optimality():
    data = rows("datasets.csv")
    s = read("summary.json")
    expected = {
        "independent_gaussian": 0,
        "correlated_gaussian": 0,
        "complementary": 14,
        "heavy_tailed": 3,
        "zero_conditional_gaussian": 1,
        "rank_two_gaussian": 0,
        "weak_conditional_gaussian": 0,
    }
    for rate in s["rates"]:
        selected = [r for r in data if r["case"] == rate["case"]]
        available = [r for r in selected if r["p_value"]]
        assert rate["n_available"] == len(available) == 25 and rate["n_missing"] == 0
        assert (
            rate["events"]
            == sum(float(r["p_value"]) <= 0.05 for r in available)
            == expected[rate["case"]]
        )
        assert rate["mean"] == rate["events"] / 25
    assert (
        "NOT species/clustering"
        in next(r for r in s["rates"] if r["case"] == "heavy_tailed")["interpretation"]
    )
    for r in rows("diagnostics.csv"):
        assert sum(int(r[f"conditional_rank_{k}"]) for k in range(5)) == 25
        assert float(r["maximum_optimality_residual"]) <= 1e-5


def test_paired_availability_and_observed_point_results():
    data = rows("paired.csv")
    s = read("paired-summary.json")
    assert (
        s["paired_datasets"],
        s["legacy_available"],
        s["current_available"],
        s["recovered"],
        s["lost"],
    ) == (100, 69, 100, 31, 0)
    assert len(data) == 100 and all(r["raw_byte_identity_verified"] == "True" for r in data)
    shared = [
        r for r in data if r["legacy_available"] == "True" and r["current_available"] == "True"
    ]
    assert len(shared) == 69
    assert all(
        abs(float(r["point_statistic_difference"])) <= 1e-8 and r["legacy_k"] == r["current_k"]
        for r in shared
    )
    assert max(float(r["nll_difference"]) for r in shared) <= 1e-5
