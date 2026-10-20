"""Protect the historical fitted-null results and independent-dataset denominators."""

import csv
import hashlib
import json
from pathlib import Path

from super_deli import __version__
from super_deli.benchmarking.config import derived_seed
from super_deli.models.empirical_config import CASES, EmpiricalStudyConfig

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/empirical-calibration/results"


def read(name):
    return json.loads((RESULTS / name).read_text())


def test_empirical_public_hashes():
    manifest = read("manifest.json")
    for path, digest in manifest["public_artifacts"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    if __version__ == manifest["reporting_version"]:
        for path, digest in manifest["reporting_code_sha256"].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest


def test_empirical_dataset_seeds_and_fit_counts():
    study = EmpiricalStudyConfig.from_dict(read("study.json"))
    with (RESULTS / "datasets.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = read("summary.json")
    count = len(CASES) * study.n_datasets
    assert len(rows) == len({(r["case"], r["repetition"]) for r in rows}) == count
    assert (
        len({r["data_seed"] for r in rows}) == len({r["null_stream_seed"] for r in rows}) == count
    )
    assert {r["data_seed"] for r in rows}.isdisjoint({r["null_stream_seed"] for r in rows})
    assert summary["datasets"] == count
    assert summary["planned_fits"] == count * (study.calibration.n_simulations + 1)
    for key in ("attempted_fits", "successful_fits", "failed_fits", "unattempted_fits"):
        assert summary[key] == sum(int(r[key]) for r in rows)
    assert summary["planned_fits"] == summary["attempted_fits"] + summary["unattempted_fits"]
    assert summary["attempted_fits"] == summary["successful_fits"] + summary["failed_fits"]
    for row in rows:
        assert int(row["data_seed"]) == derived_seed(
            study.random_seed, row["case"], int(row["repetition"]), "observed_data"
        )
        assert int(row["null_stream_seed"]) == derived_seed(
            study.random_seed, row["case"], int(row["repetition"]), "null_stream"
        )


def test_empirical_rates_and_stress_interpretation():
    with (RESULTS / "datasets.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    study = read("study.json")
    for rate in read("summary.json")["rates"]:
        selected = [r for r in rows if r["case"] == rate["case"]]
        available = [r for r in selected if r["p_value"]]
        events = sum(float(r["p_value"]) <= study["calibration"]["alpha"] for r in available)
        assert len(selected) == study["n_datasets"]
        assert rate["n_available"] == len(available) and rate["n_missing"] == len(selected) - len(
            available
        )
        assert rate["events"] == events and rate["mean"] == events / len(available)
    assert "NOT species/clustering" in read("summary.json")["rates"][-1]["interpretation"]
