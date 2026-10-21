"""Protect the published frozen genomic study and uncertainty denominators."""

import csv
import hashlib
import json
from pathlib import Path

from super_deli import __version__
from super_deli.benchmarking.config import derived_seed
from super_deli.models.genomic_config import CASES, GenomicStudyConfig

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/genomic-calibration/results"


def read(name):
    return json.loads((RESULTS / name).read_text())


def rows(name):
    with (RESULTS / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_public_hashes_and_frozen_reporting_source_equality():
    m = read("manifest.json")
    for path, digest in m["public_artifacts"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    if m["reporting_version"] == __version__:
        for path, digest in m["reporting_code_sha256"].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert m["frozen_reporting_python_sources_equal"]
    assert m["audit_counts"] == {
        "datasets": 125,
        "archive_members": 252250,
        "verified_successful_pipelines": 12500,
        "raw_files": 15055,
    }


def test_dataset_seeds_and_planned_attempt_denominators():
    study = GenomicStudyConfig.from_dict(read("study.json"))
    data = rows("datasets.csv")
    s = read("summary.json")
    assert len(data) == len(CASES) * study.n_datasets == 125
    assert len({(r["case"], r["repetition"]) for r in data}) == 125
    observed, null = set(), set()
    for r in data:
        i = int(r["repetition"])
        assert int(r["data_seed"]) == derived_seed(study.random_seed, r["case"], i, "observed_data")
        assert int(r["null_stream_seed"]) == derived_seed(
            study.random_seed, r["case"], i, "null_stream"
        )
        observed.add(r["data_seed"])
        null.add(r["null_stream_seed"])
        assert int(r["planned_fits"]) == int(r["successful_fits"]) == 100
        assert int(r["failed_fits"]) == int(r["unattempted_fits"]) == 0
    assert len(observed) == len(null) == 125 and observed.isdisjoint(null)
    assert s["successful_fits"] == s["attempted_fits"] == s["planned_fits"] == 12500


def test_rate_events_availability_and_conditional_tail_exports():
    expected = dict(zip(CASES, (1, 0, 25, 0, 25), strict=True))
    data = rows("dataset-diagnostics.csv")
    assert len(data) == 125
    for rate in read("summary.json")["rates"]:
        selected = [r for r in data if r["case"] == rate["case"]]
        assert rate["n_available"] == len(selected) == 25 and rate["n_missing"] == 0
        assert (
            sum(float(r["p_value"]) <= 0.05 for r in selected)
            == rate["events"]
            == expected[rate["case"]]
        )
        assert rate["mean"] == rate["events"] / 25
        assert rate["wilson_95"][0] <= rate["mean"] <= rate["wilson_95"][1]
        for r in selected:
            assert float(r["p_value"]) == (1 + int(r["exceedances"])) / 100
            assert float(r["minimum_p_value"]) == 0.01
            assert (
                0
                <= float(r["conditional_tail_95_low"])
                <= float(r["conditional_tail_95_high"])
                <= 1
            )


def test_selected_k_boundary_and_missingness_diagnostics_agree():
    data = rows("dataset-diagnostics.csv")
    diagnostics = {r["case"]: r for r in rows("diagnostics.csv")}
    for case, d in diagnostics.items():
        selected = [r for r in data if r["case"] == case]
        for k in range(1, 5):
            assert int(d[f"point_k{k}"]) == sum(int(r["selected_k"]) == k for r in selected)
        assert int(d["point_t_zero"]) == sum(float(r["statistic"]) == 0 for r in selected)
        assert int(d["total_boundary_loci"]) == sum(int(r["boundary_loci"]) for r in selected)
        assert int(d["total_missing_calls"]) == sum(int(r["missing_calls"]) for r in selected)
    assert int(diagnostics["rare_missing_hwe"]["total_boundary_loci"]) == 92
    assert int(diagnostics["rare_missing_hwe"]["total_missing_calls"]) == 20408
    assert int(diagnostics["rare_missing_hwe"]["point_k2"]) == 11
    assert int(diagnostics["cline_stress"]["point_k3"]) == 11
    assert int(diagnostics["cline_stress"]["point_k4"]) == 14
