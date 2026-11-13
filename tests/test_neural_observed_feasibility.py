"""Full invented source/benchmark lifecycle; archived heads train only as fixtures."""

import json
import subprocess
import sys

import pytest

pytest.importorskip("torch")


@pytest.fixture(scope="module")
def complete(tmp_path_factory):
    from super_deli.benchmarking.observed_feasibility_fixture import smoke

    out = tmp_path_factory.mktemp("invented_observed") / "smoke"
    result = smoke(out, workers=1)
    assert result["passed"] and result["invented_only"]
    return out, result


def test_complete_scientific_budget_and_guarded_installation(complete):
    out, result = complete
    summary = result["study"]
    assert summary["source_heads_retrained"] == 0 and summary["main_detector_fits"] == 2
    assert summary["main_trees"] == 200 and summary["preflight_trees"] == 200
    assert summary["primary_rates"] == 8 and summary["paired_contrasts"] == 6
    assert not summary["default_changes"] and not summary["pvalue_changes"]
    r = json.loads((out / "study/main-execution.json").read_text())
    assert r["completed"] and r["attempted"] == r["planned"] == 4 and r["neural_fits"] == 0


def test_manifest_bound_source_tamper_is_rejected(complete, tmp_path):
    out, _ = complete
    from super_deli.benchmarking.observed_feasibility_source import check_source

    binding = json.loads((out / "study/replay-binding.json").read_text())
    binding["source_sha256"]["fixture.json"] = "0" * 64
    with pytest.raises(ValueError):
        check_source(out / "invented-source", binding)


def test_completed_native_scores_are_not_replaced_on_resume(complete):
    out, _ = complete
    root = out / "study"
    before = (root / "main-execution.json").read_bytes()
    command = [
        sys.executable,
        "-m",
        "super_deli",
        "benchmark-observed-feasibility",
        "run",
        "--archive-root",
        str(out / "invented-source"),
        "--output-dir",
        str(root),
        "--phase",
        "main",
        "--fixture",
    ]
    r = subprocess.run(command, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (root / "main-execution.json").read_bytes() == before


def test_source_and_scientific_config_fixture_mismatch_is_rejected(complete):
    out, _ = complete
    from super_deli.benchmarking.observed_feasibility_config import ObservedFeasibilityStudy
    from super_deli.benchmarking.observed_feasibility_study import jobs

    binding = json.loads((out / "study/replay-binding.json").read_text())
    with pytest.raises(ValueError):
        jobs(ObservedFeasibilityStudy(), "main", binding)
