"""Invented numerical cases, legacy parity, complete gates and corrupt-evidence detection."""
# ruff: noqa: E402

import json
import shutil

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")
from super_deli.benchmarking.gaussian_graph_audit import audit_phase
from super_deli.benchmarking.gaussian_graph_config import StudyConfig
from super_deli.benchmarking.gaussian_graph_fit import (
    assign,
    fit_gaussian,
    legacy_selection,
    observed_graph,
    tail_decision,
)
from super_deli.benchmarking.gaussian_graph_reference import gaussian_audit
from super_deli.benchmarking.gaussian_graph_report import report
from super_deli.benchmarking.gaussian_graph_study import (
    check_frozen,
    freeze,
    gate,
    inventory,
    run_phase,
)
from super_deli.benchmarking.unknown_k_data import model_config
from super_deli.io.config import read_json
from super_deli.models.spectral_graph import fit_scores


def test_psd_fit_preserves_rank_without_variance_floor():
    z = np.random.default_rng(99011).normal(size=(40, 2))
    x = z @ np.array([[1, 0, 1, 2, 3, 1, -1, 2], [0, 1, 1, -1, 2, 4, 1, 2]])
    a = fit_gaussian(x)
    assert a["effective_rank"] == 2 and a["factor"].shape == (8, 2)
    gaussian_audit(x, a)
    for bad in (np.ones((40, 8)), np.arange(40.0)[:, None] @ np.ones((1, 8)), x[:12], x * np.nan):
        with pytest.raises(ValueError):
            fit_gaussian(bad)


def profiles():
    pilot = np.tile(np.linspace(0.01, 0.2, 49)[:, None], (1, 5))
    reference = np.ones((99, 5)) * 0.1
    return pilot, reference


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4])
def test_known_tail_and_count_proposal(index):
    p, r = profiles()
    observed = p.mean(0)
    observed[index] += 2
    result, a = tail_decision(observed, p, r)
    assert result["p_reference"] == 0.01 and result["exceedances"] == 0
    assert result["proposed_k"] == index + 2
    assert result["reason"] == ("search_boundary" if index == 4 else None)
    assert a["z"].argmax() == index


def test_no_split_ties_degenerate_and_incomplete_reference():
    p, r = profiles()
    result, _ = tail_decision(p.mean(0), p, r)
    assert result["p_reference"] == 0.01  # Means .105 exceed fixed-reference .1.
    assert result["reason"] == "unresolved_gap_choice"
    result, _ = tail_decision(np.zeros(5), p, r)
    assert result["p_reference"] == 1 and result["proposed_k"] == 1
    for pilot, reference, reason in (
        (p[:48], r, "incomplete_reference_draws"),
        (p, r[:98], "incomplete_reference_draws"),
        (p * 0, r, "degenerate_pilot_normalizer"),
        (p, r * np.nan, "nonfinite_reference_draws"),
    ):
        result, _ = tail_decision(np.zeros(5), pilot, reference)
        assert result["status"] == "withheld" and result["p_reference"] is None
        assert result["reason"] == reason and result["selected_k"] is None


@pytest.mark.parametrize("neighbors", [5, 15])
def test_legacy_shared_graph_parity(neighbors):
    x = np.random.default_rng([99013, neighbors]).normal(size=(40, 8))
    ids = [f"invented_{i:04d}" for i in range(40)]
    affinity, a, info = observed_graph(x, ids, neighbors)
    selection, gaps = legacy_selection(a["eigenvalues"])
    actual, evidence = assign(affinity, a, info, selection, neighbors)
    evidence["gaps"] = gaps
    expected, direct = fit_scores(
        (x - x.mean(0)) / x.std(0), ids, model_config(f"spectral_{neighbors}")
    )
    assert actual == {k: v for k, v in expected.items() if k != "seconds"}
    for key in direct:
        np.testing.assert_array_equal(direct[key], evidence[key])


def test_no_split_and_failed_candidate_do_not_fall_back():
    x = np.random.default_rng(99014).normal(size=(40, 8))
    ids = [f"invented_{i:04d}" for i in range(40)]
    affinity, a, info = observed_graph(x, ids, 5)
    r, b = assign(affinity, a, {**info, "components": 3}, {"proposed_k": 2, "reason": None}, 5)
    assert r["status"] == "withheld" and "labels" not in b and r["selected_k"] is None
    r, b = assign(
        affinity,
        a,
        {**info, "components": 3},
        {"proposed_k": 1, "reason": None, "interpretation": "no_reference_supported_split"},
        5,
    )
    assert r["selected_k"] == 1 and np.all(b["labels"] == 0)


def test_overflowing_normalized_statistic_is_withheld():
    pilot = np.tile(np.linspace(0, 1e-10, 49)[:, None], (1, 5))
    with np.errstate(over="ignore"):
        result, _ = tail_decision(np.full(5, 1e308), pilot, np.zeros((99, 5)))
    assert result["p_reference"] is None and result["reason"] == "nonfinite_reference_statistic"
    assert result["selected_k"] is None


@pytest.fixture(scope="module")
def fixture_study(tmp_path_factory):
    root = tmp_path_factory.mktemp("gaussian-graph-fixture")
    (root / "approved-plan.md").write_text("Invented engineering fixture; not performance evidence")
    freeze(root, StudyConfig(fixture=True))
    run_phase(root, "preflight", 1)
    audit_phase(root, "preflight")
    assert gate(root, 0)["passed"]
    return root


def test_complete_protocol_and_report(fixture_study):
    root = fixture_study
    assert run_phase(root, "main", 2)["evaluations"] == 1800
    result = audit_phase(root, "main")
    assert result["actual_evaluation_attempts"] == 1800
    assert report(root)["fixture"]
    aggregate = read_json(root / "aggregate.json")
    assert aggregate["point_outputs"] == 24
    assert len(aggregate["primary_rates"]) == 36 and len(aggregate["paired_comparisons"]) == 12
    with pytest.raises(FileExistsError):
        run_phase(root, "main", 1)


def test_failed_cap_submits_nothing(tmp_path):
    (tmp_path / "approved-plan.md").write_text("Engineering cap fixture")
    freeze(tmp_path, StudyConfig(fixture=True, artifact_cap_bytes=1))
    with pytest.raises(ValueError, match="Artifact cap"):
        run_phase(tmp_path, "preflight", 1)
    assert not (tmp_path / "preflight").exists()


def test_freeze_and_ledger_tampering(tmp_path, fixture_study):
    root = tmp_path / "copy"
    shutil.copytree(fixture_study, root)
    path = root / "preflight-execution.json"
    data = read_json(path)
    data["datasets"][0]["tables"][0]["seconds"] = 0
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="ledger"):
        gate(root, 0)
    with pytest.raises(ValueError, match="gate"):
        run_phase(root, "main", 1)
    (root / "approved-plan.md").write_text("Changed plan")
    with pytest.raises(ValueError, match="Frozen"):
        check_frozen(root)


def test_retained_tail_mutation_detected_even_after_inventory_update(tmp_path, fixture_study):
    root = tmp_path / "copy"
    shutil.copytree(fixture_study, root)
    folder = next((root / "preflight").glob("*/clean/point"))
    path = folder / "reference_5/summary.json"
    original = read_json(path)
    original["p_reference"] = 0.123
    path.write_text(json.dumps(original))
    # An updated inventory alone cannot make corrupted numerical evidence valid.
    (root / "preflight-inventory.json").write_text(json.dumps(inventory(root / "preflight")))
    with pytest.raises(ValueError, match="summary"):
        audit_phase(root, "preflight")
