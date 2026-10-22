"""Protect the committed historical calibration evidence and its denominators."""

import csv
import hashlib
import json
from pathlib import Path

from super_deli import __version__

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/calibration/results"


def read(name):
    return json.loads((RESULTS / name).read_text())


def test_calibration_public_artifact_hashes():
    manifest = read("manifest.json")
    for relative, digest in manifest["public_artifacts"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    if __version__ == manifest["reporting_version"]:
        for relative, digest in manifest["reporting_code_sha256"].items():
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest


def test_frozen_calibration_dataset_identity_and_draw_denominators():
    with (RESULTS / "datasets.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary, study = read("summary.json"), read("study.json")
    expected = 4 * study["n_repetitions"]
    assert len(rows) == len({(r["scenario"], r["repetition"]) for r in rows}) == expected
    assert len({r["data_seed"] for r in rows}) == expected
    assert len({r["calibration_seed"] for r in rows}) == expected
    assert {r["data_seed"] for r in rows}.isdisjoint({r["calibration_seed"] for r in rows})
    assert summary["datasets"] == expected
    assert summary["planned_null_draws"] == expected * study["calibration"]["n_simulations"]
    assert summary["attempted_null_draws"] == (
        summary["successful_null_draws"] + summary["failed_null_draws"]
    )
    assert summary["planned_null_draws"] == (
        summary["attempted_null_draws"] + summary["unattempted_null_draws"]
    )


def test_published_rates_reconstruct_from_available_dataset_events():
    with (RESULTS / "datasets.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    study = read("study.json")
    for rate in read("summary.json")["rates"]:
        scenario = [r for r in rows if r["scenario"] == rate["scenario"]]
        available = [r for r in scenario if r["p_value"]]
        rejected = sum(float(r["p_value"]) <= study["calibration"]["alpha"] for r in available)
        assert rate["n_available"] == len(available)
        assert rate["n_missing"] == len(scenario) - len(available)
        assert rate["events"] == rejected
        assert rate["mean"] == rejected / len(available)
    assert "NOT type-I" in read("summary.json")["rates"][-1]["interpretation"]
