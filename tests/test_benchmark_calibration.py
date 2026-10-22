"""Independent score generators and tiny end-to-end validation artifacts."""

# ruff: noqa: E402

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from super_deli.benchmarking.score_calibration import SCENARIOS, run_score_study, simulate_scores
from super_deli.io.config import read_json

STUDY = Path(__file__).resolve().parents[1] / "examples/calibration/study.json"


def test_generators_are_independent_and_reconstructable():
    for scenario in SCENARIOS:
        first = simulate_scores(scenario, 64, 13)
        np.testing.assert_array_equal(first, simulate_scores(scenario, 64, 13))
        assert not np.array_equal(first, simulate_scores(scenario, 64, 14))
    correlated = simulate_scores("correlated_gaussian", 50000, 71)
    np.testing.assert_allclose(np.cov(correlated, rowvar=False), [[4, 1.6], [1.6, 1]], rtol=0.03)


def test_tiny_score_study_uses_new_seeds_and_full_denominators(tmp_path):
    options = read_json(STUDY)
    options["n_repetitions"] = 2
    options["calibration"]["n_simulations"] = 2
    options["search"]["n_init"] = 1
    result = run_score_study(options, output_dir=tmp_path / "study", scenario="spherical_gaussian")
    assert result["requested"] == 2 and result["attempted_null_draws"] == 4
    rows = result["datasets"]
    assert rows[0]["data_seed"] != rows[1]["data_seed"]
    assert rows[0]["calibration_seed"] != rows[0]["data_seed"]
    assert json.loads((tmp_path / "study/summary.json").read_text()) == result
    assert "Type-I" in result["rate_interpretation"]


@pytest.fixture
def completed_study(tmp_path):
    options = read_json(STUDY)
    options["n_repetitions"] = 1
    options["calibration"]["n_simulations"] = 2
    options["search"]["n_init"] = 1
    root = tmp_path / "shards"
    root.mkdir()
    for scenario in SCENARIOS:
        run_score_study(options, output_dir=root / scenario, scenario=scenario)
    return root


def test_independent_audit_and_output_safety(completed_study, tmp_path):
    from super_deli.benchmarking.calibration_analysis import audit_score_study

    output = tmp_path / "audit"
    summary = audit_score_study(completed_study, output_dir=output)
    assert summary["datasets"] == 4 and summary["successful_null_draws"] == 8
    assert summary["failed_null_draws"] == 0
    with pytest.raises(FileExistsError):
        audit_score_study(completed_study, output_dir=output)
    with pytest.raises(ValueError, match="outside"):
        audit_score_study(completed_study, output_dir=completed_study / "audit")


def test_auditor_catches_draw_tampering_even_with_updated_hashes(completed_study, tmp_path):
    from super_deli.benchmarking.calibration_analysis import audit_score_study
    from super_deli.core.provenance import fingerprint_file

    shard = completed_study / SCENARIOS[0]
    directory = shard / "rep_0000"
    log_path = directory / "null_replicates.json"
    log = read_json(log_path)
    log[0]["draw_seed"] += 1
    log_path.write_text(json.dumps(log))
    result_path = directory / "summary.json"
    result = read_json(result_path)
    result["artifacts"]["null_replicates.json"] = fingerprint_file(log_path).sha256
    result_path.write_text(json.dumps(result))
    summary_path = shard / "summary.json"
    summary = read_json(summary_path)
    summary["datasets"][0]["summary_sha256"] = fingerprint_file(result_path).sha256
    summary_path.write_text(json.dumps(summary))
    (shard / "ledger_0000.json").write_text(json.dumps(summary["datasets"]))
    with pytest.raises(ValueError, match="draw or seed"):
        audit_score_study(completed_study, output_dir=tmp_path / "invalid")


def test_auditor_preserves_failed_point_denominators(tmp_path, monkeypatch):
    from super_deli.benchmarking.calibration_analysis import audit_score_study
    from super_deli.models import gaussian_calibration

    original = gaussian_calibration.calibrate_gaussian_scores
    calls = []

    def fail_first(*args, **kwargs):
        calls.append(None)
        if len(calls) == 1:
            raise ValueError("injected point failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(gaussian_calibration, "calibrate_gaussian_scores", fail_first)
    options = read_json(STUDY)
    options["n_repetitions"] = 1
    options["calibration"]["n_simulations"] = 2
    options["search"]["n_init"] = 1
    root = tmp_path / "shards"
    root.mkdir()
    for scenario in SCENARIOS:
        run_score_study(options, output_dir=root / scenario, scenario=scenario)
    result = audit_score_study(root, output_dir=tmp_path / "audit")
    assert result["point_failures"] == 1 and result["datasets"] == 4
    assert result["planned_null_draws"] == 8
    assert result["attempted_null_draws"] == 6 and result["unattempted_null_draws"] == 2
    assert result["rates"][0]["n_missing"] == 1
    assert result["rates"][0]["wilson_95"] is None
