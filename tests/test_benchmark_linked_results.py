"""Protect frozen linked-study evidence, block support and paired outcomes."""

import csv
import hashlib
import json
from pathlib import Path

from super_deli import __version__
from super_deli.benchmarking.config import derived_seed
from super_deli.models.linked_config import CASES, LinkedStudyConfig

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/linked-calibration/results"


def read(name):
    return json.loads((RESULTS / name).read_text())


def rows(name):
    with (RESULTS / name).open(newline="") as f:
        return list(csv.DictReader(f))


def test_public_artifacts_frozen_code_and_legacy_inventory():
    m = read("manifest.json")
    assert "local_artifact_root" not in m
    assert m["local_artifact_id"] == "linked-null-20261021"
    runtime = read("runtime_versions.json")
    assert all(set(v) == {"version"} for v in runtime["libraries"].values())
    assert "/Users/" not in json.dumps(m) + json.dumps(runtime)
    for p, digest in m["public_artifacts"].items():
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
    if m["reporting_version"] == __version__:
        for p, digest in m["reporting_code_sha256"].items():
            assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
    assert m["frozen_reporting_python_sources_equal"]
    legacy = json.loads((ROOT / "examples/genomic-calibration/results/manifest.json").read_text())
    assert m["legacy_raw_files_inventory_sha256"] == legacy["raw_files_inventory_sha256"]
    assert m["audit_counts"] == {
        "datasets": 150,
        "archive_members": 329400,
        "verified_successful_pipelines": 15000,
        "raw_files": 18036,
    }


def test_independent_dataset_seeds_and_attempt_denominators():
    study = LinkedStudyConfig.from_dict(read("study.json"))
    data = rows("datasets.csv")
    s = read("summary.json")
    assert len(data) == len(CASES) * study.n_datasets == 150
    assert len({(r["case"], r["repetition"]) for r in data}) == 150
    observed, null = set(), set()
    for r in data:
        i = int(r["repetition"])
        assert int(r["data_seed"]) == derived_seed(study.random_seed, r["case"], i, "observed_data")
        assert int(r["null_stream_seed"]) == derived_seed(
            study.random_seed, r["case"], i, "null_stream"
        )
        observed.add(r["data_seed"])
        null.add(r["null_stream_seed"])
        assert (
            int(r["planned_fits"]) == int(r["attempted_fits"]) == int(r["successful_fits"]) == 100
        )
        assert int(r["failed_fits"]) == int(r["unattempted_fits"]) == 0
    assert len(observed) == len(null) == 150 and observed.isdisjoint(null)
    assert s["planned_fits"] == s["attempted_fits"] == s["successful_fits"] == 15000
    assert s["failed_fits"] == s["unattempted_fits"] == 0


def test_event_rates_and_conditional_tail_intervals():
    expected = {
        "common_hwe": 1,
        "linked_stress": 0,
        "moderate_ld": 0,
        "four_groups": 25,
        "cline_stress": 25,
        "split_block_stress": 1,
    }
    data = rows("dataset-diagnostics.csv")
    assert len(data) == 150
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
    borderline = [
        r for r in data if r["case"] == "split_block_stress" and float(r["p_value"]) <= 0.05
    ]
    assert len(borderline) == 1 and float(borderline[0]["p_value"]) == 0.05
    assert (
        float(borderline[0]["conditional_tail_95_low"])
        < 0.05
        < float(borderline[0]["conditional_tail_95_high"])
    )


def test_block_support_diagnostics_and_selected_k():
    blocks = rows("block-diagnostics.csv")
    data = rows("dataset-diagnostics.csv")
    assert len(blocks) == 2400
    for r in data:
        bs = [b for b in blocks if (b["case"], b["repetition"]) == (r["case"], r["repetition"])]
        assert len(bs) == len({b["block_id"] for b in bs}) == 16
        assert int(r["total_profiles"]) == sum(int(b["profiles"]) for b in bs)
        for b in bs:
            assert int(b["loci"]) == 8 and 1 <= int(b["profiles"]) <= 64
            assert 0 <= int(b["singleton_profiles"]) <= int(b["profiles"])
            assert float(b["singleton_specimen_fraction"]) == int(b["singleton_profiles"]) / 64
            assert 1 / int(b["profiles"]) <= float(b["maximum_profile_probability"]) <= 1
            if b["case"] in {"linked_stress", "split_block_stress"}:
                assert int(b["profiles"]) <= 3
    for d in rows("diagnostics.csv"):
        selected = [r for r in data if r["case"] == d["case"]]
        for k in range(1, 5):
            assert int(d[f"point_k{k}"]) == sum(int(r["selected_k"]) == k for r in selected)
        assert int(d["point_t_zero"]) == sum(float(r["statistic"]) == 0 for r in selected)
    common = next(d for d in rows("diagnostics.csv") if d["case"] == "common_hwe")
    assert (
        float(common["profile_count_median"]) == 62
        and float(common["mean_singleton_specimen_fraction"]) > 0.92
    )


def test_byte_paired_point_statistics_and_distinct_null_scope():
    s = read("paired-summary.json")
    pairs = rows("paired.csv")
    assert (
        s["paired_datasets"] == s["legacy_available"] == s["current_available"] == len(pairs) == 100
    )
    assert s["recovered"] == s["lost"] == 0 and s["raw_byte_identity_verified"]
    assert "different hypotheses" in s["scope"] and "different null seed namespaces" in s["scope"]
    assert all(
        r["raw_byte_identity_verified"] == "True"
        and float(r["point_statistic_difference"]) == 0
        and r["legacy_k"] == r["current_k"]
        for r in pairs
    )
    assert all(r["legacy_p_value"] == r["current_p_value"] for r in pairs)
