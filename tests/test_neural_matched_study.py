"""Engineering-only complete lifecycle, full optimizer references and evidence gates."""
# ruff: noqa: E402

import json
import shutil

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("sklearn")
from super_deli.benchmarking.matched_audit import audit_phase
from super_deli.benchmarking.matched_config import CASES, StudyConfig
from super_deli.benchmarking.matched_data import generate
from super_deli.benchmarking.matched_reference import reference_data
from super_deli.benchmarking.matched_report import report
from super_deli.benchmarking.matched_study import check_frozen, freeze, gate, run_phase
from super_deli.io.config import read_json


def test_independent_laws():
    for index, case in enumerate(CASES):
        task = {
            "case": case,
            "n": 32,
            "data_seed": [109021, index],
            "noise_seed": [109023, index],
            "permutation_seed": [109027, index],
            "ecology_order_seed": [109031, index],
        }
        actual, expected = generate(task), reference_data(task)
        assert set(actual) == set(expected)
        for key in actual:
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-11)


@pytest.fixture(scope="module")
def fixture_study(tmp_path_factory):
    root = tmp_path_factory.mktemp("matched-neural-fixture")
    (root / "approved-plan.md").write_text(
        "Invented engineering fixture; not scientific validation"
    )
    freeze(root, StudyConfig(fixture=True))
    run_phase(root, "preflight", 1)
    audit_phase(root, "preflight")
    assert gate(root, 0)["passed"]
    return root


def test_full_gated_protocol(fixture_study):
    root = fixture_study
    assert run_phase(root, "main", 2)["workflows"] == 83
    audit = audit_phase(root, "main")
    assert audit["actual_workflow_attempts"] == 83 and audit["numpy_optimizer_replays"] == 63
    assert audit["optimizer_updates_checked"] == 31500 and audit["successful_point_replays"] == 14
    result = report(root)
    assert (
        result["point_outputs"] == 14
        and result["primary_rates"] == 28
        and result["paired_comparisons"] == 18
    )
    with pytest.raises(FileExistsError):
        run_phase(root, "main", 1)


def test_budget_and_ledger_tampering(tmp_path, fixture_study):
    root = tmp_path / "copy"
    shutil.copytree(fixture_study, root)
    (root / "storage-runtime-gate.json").rename(root / "passed-gate.json")
    assert not gate(root, 2**31)["passed"]
    with pytest.raises(ValueError, match="gate"):
        run_phase(root, "main", 1)
    path = root / "preflight-execution.json"
    data = read_json(path)
    data["datasets"][0]["records"][0]["seconds"] = 0
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="ledger"):
        gate(root, 0)
    (root / "approved-plan.md").write_text("Changed plan")
    with pytest.raises(ValueError, match="Frozen"):
        check_frozen(root)


def test_corrupted_result_inventory_detected(tmp_path, fixture_study):
    root = tmp_path / "copy"
    shutil.copytree(fixture_study, root)
    p = next((root / "preflight").rglob("point.npz"))
    p.write_bytes(p.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="inventory"):
        audit_phase(root, "preflight")


def test_gated_per_step_verification_and_source_audit(tmp_path, fixture_study):
    from super_deli.benchmarking.matched_step_study import (
        descriptors,
        probe_grid,
        run,
    )
    from super_deli.benchmarking.matched_step_study import gate as step_gate

    root = tmp_path / "copy"
    shutil.copytree(fixture_study, root)
    config = StudyConfig(fixture=True)
    probes = probe_grid(root, config)
    assert len(probes) == 6
    result = run(root, "step-preflight", probes, 1, config)
    assert result["passed"] and result["updates_checked"] == 3000
    assert step_gate(root, 0, 0)["passed"]
    result = run(root, "step-preflight-verification", descriptors(root, "preflight"), 1, config)
    assert result["passed"] and result["verification_replays"] == 63
    (root / "preflight-audit.json").rename(root / "retained-serial-audit.json")
    (root / "preflight-training-audit.json").rename(root / "retained-serial-training-audit.json")
    audit = audit_phase(root, "preflight")
    assert audit["passed"] and audit["training_audit_mode"] == "actual_path_per_step"
    assert audit["optimizer_updates_checked"] == 31500
