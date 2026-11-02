"""Separate invented protocol, baseline instrumentation parity and resource guards."""
# ruff: noqa: E402

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from sklearn.mixture import GaussianMixture

from super_deli.benchmarking.neural_analysis import arrays
from super_deli.benchmarking.neural_study import write_dataset
from super_deli.benchmarking.unknown_k_audit import audit_phase
from super_deli.benchmarking.unknown_k_config import StudyConfig, tasks
from super_deli.benchmarking.unknown_k_data import generate, model_config
from super_deli.benchmarking.unknown_k_reference import audit_candidates, reference
from super_deli.benchmarking.unknown_k_report import report
from super_deli.benchmarking.unknown_k_run import run_arm
from super_deli.benchmarking.unknown_k_study import check_frozen, freeze, gate, run_phase
from super_deli.io.config import read_json
from super_deli.workflows.tabular_gmm import run_tabular_gmm


def test_all_source_laws_on_engineering_seeds():
    # Explicitly not the future scientific preflight/main random streams.
    for task in tasks(StudyConfig(), "preflight"):
        task = {
            **task,
            "data_seed": [99071, task["case"].__len__(), task["n"]],
            "noise_seed": [99072, task["n"]],
            "permutation_seed": [99073, task["n"]],
        }
        actual, expected = generate(task), reference(task)
        assert set(actual) == set(expected)
        for k in actual:
            np.testing.assert_allclose(actual[k], expected[k], atol=1e-11)


def test_instrumented_baseline_is_identical(tmp_path):
    x = np.random.default_rng(99074).normal(size=(40, 8))
    raw = tmp_path / "raw"
    ds = write_dataset(x, raw)
    original = GaussianMixture.fit
    direct = run_tabular_gmm(
        ds, model_config=model_config("gmm"), base_dir=raw, output_dir=tmp_path / "direct"
    )
    row = run_arm(ds, raw, tmp_path / "wrapped", "gmm", "point")
    assert row["status"] == "available" and row["selected_k"] == direct["selected_k"]
    assert GaussianMixture.fit is original
    for name in ("mixture.npz", "preprocessing.npz"):
        a, b = arrays(tmp_path / "direct" / name), arrays(tmp_path / "wrapped" / name)
        assert set(a) == set(b)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    audit_candidates(tmp_path / "wrapped", len(x))
    trace = arrays(tmp_path / "wrapped/candidate-trace.npz")
    assert all(f"k{k}_covariances" in trace for k in range(1, 5))


@pytest.fixture(scope="module")
def fixture_study(tmp_path_factory):
    root = tmp_path_factory.mktemp("unknown-k-fixture")
    (root / "approved-plan.md").write_text(
        "Invented engineering fixture, not scientific performance"
    )
    config = StudyConfig(fixture=True)
    freeze(root, config)
    run_phase(root, "preflight", 1)
    audit_phase(root, "preflight")
    assert gate(root, 0)["passed"]
    return root


def test_complete_gated_protocol(fixture_study):
    root = fixture_study
    assert read_json(root / "preflight-audit.json")["evaluations"] == 74
    assert run_phase(root, "main", 2)["evaluations"] == 64
    result = audit_phase(root, "main")
    assert result["passed"] and result["exact_workflow_replays"] == 16
    assert report(root)["fixture"]
    aggregate = read_json(root / "aggregate.json")
    assert aggregate["point_workflows"] == 48 and len(aggregate["primary_rates"]) == 80
    assert sum(aggregate["statuses"].values()) == 48
    with pytest.raises(FileExistsError):
        run_phase(root, "main", 1)


def test_frozen_integrity_and_failed_gate(tmp_path, fixture_study):
    import shutil

    root = tmp_path / "bad"
    shutil.copytree(fixture_study, root)
    # Keep every preflight result but recompute only a deliberately failed budget gate.
    (root / "storage-runtime-gate.json").rename(root / "original-passed-gate.json")
    assert not gate(root, 2 * 1024**3)["passed"]
    with pytest.raises(ValueError, match="gate"):
        run_phase(root, "main", 1)
    (root / "approved-plan.md").write_text("changed after freeze")
    with pytest.raises(ValueError, match="Frozen"):
        check_frozen(root)


def test_inventory_corruption_is_detected(tmp_path, fixture_study):
    import shutil

    root = tmp_path / "bad"
    shutil.copytree(fixture_study, root)
    path = next((root / "preflight").rglob("*.npz"))
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="inventory"):
        audit_phase(root, "preflight")


def test_execution_ledger_is_bound_to_resource_gate(tmp_path, fixture_study):
    import json
    import shutil

    root = tmp_path / "bad-ledger"
    shutil.copytree(fixture_study, root)
    path = root / "preflight-execution.json"
    data = read_json(path)
    data["probe_fits"][0]["seconds"] = 0.0
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="ledger changed"):
        gate(root, 0)
    with pytest.raises(ValueError, match="gate"):
        run_phase(root, "main", 1)


def test_failed_initial_cap_submits_no_work(tmp_path):
    root = tmp_path / "bounded"
    root.mkdir()
    (root / "approved-plan.md").write_text("Invented resource fixture")
    freeze(root, StudyConfig(fixture=True, artifact_cap_bytes=1))
    with pytest.raises(ValueError, match="Artifact cap"):
        run_phase(root, "preflight", 2)
    assert not (root / "preflight").exists()
