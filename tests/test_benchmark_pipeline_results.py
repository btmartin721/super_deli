"""Protect published reference-law evidence, identities and fit denominators."""

import csv
import hashlib
import json
from pathlib import Path

from super_deli import __version__
from super_deli.benchmarking.config import derived_seed
from super_deli.models.pipeline_config import CASES, PipelineStudyConfig

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/pipeline-calibration/results"


def read(name):
    return json.loads((RESULTS / name).read_text())


def test_reference_public_evidence_checksums():
    manifest = read("manifest.json")
    for path, digest in manifest["public_artifacts"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    if manifest["reporting_version"] == __version__:
        for path, digest in manifest["reporting_code_sha256"].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest


def test_reference_dataset_seeds_and_denominators():
    study = PipelineStudyConfig.from_dict(read("study.json"))
    summary = read("summary.json")
    with (RESULTS / "datasets.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = len(CASES) * study.n_datasets
    assert len(rows) == len({(r["case"], r["repetition"]) for r in rows}) == expected
    assert (
        len({r["data_seed"] for r in rows})
        == len({r["null_stream_seed"] for r in rows})
        == expected
    )
    assert {r["data_seed"] for r in rows}.isdisjoint({r["null_stream_seed"] for r in rows})
    assert summary["datasets"] == expected
    assert summary["planned_fits"] == expected * (study.calibration.n_simulations + 1)
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


def test_reference_rates_reconstruct_from_independent_datasets():
    study = read("study.json")
    with (RESULTS / "datasets.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for rate in read("summary.json")["rates"]:
        case = [r for r in rows if r["case"] == rate["case"]]
        available = [r for r in case if r["p_value"]]
        rejected = sum(float(r["p_value"]) <= study["calibration"]["alpha"] for r in available)
        assert len(case) == study["n_datasets"]
        assert rate["n_available"] == len(available)
        assert rate["n_missing"] == len(case) - len(available)
        assert rate["events"] == rejected
        assert rate["mean"] == rejected / len(available)
