"""Untrained invented fixtures, independent graph audits and failure guards."""

# Optional numerical dependencies must be checked before importing their users.
# ruff: noqa: E402

import copy

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from super_deli.benchmarking.view_signal_archive import bind, check_binding
from super_deli.benchmarking.view_signal_audit import audit_fit
from super_deli.benchmarking.view_signal_config import REPRESENTATIONS, ViewSignalConfig
from super_deli.benchmarking.view_signal_coordinates import load_context, native_arm
from super_deli.benchmarking.view_signal_fit import fit_coordinates
from super_deli.benchmarking.view_signal_fixture import create_fixture
from super_deli.benchmarking.view_signal_metrics import (
    branch_reference,
    geometry,
    linear_recovery,
    neighbors,
)
from super_deli.benchmarking.view_signal_report import report
from super_deli.benchmarking.view_signal_study import (
    audit_phase,
    check_frozen,
    freeze,
    gate,
    run_phase,
)
from super_deli.io.config import read_json


@pytest.fixture
def invented(tmp_path):
    source = create_fixture(tmp_path / "source")
    out = tmp_path / "diagnostic"
    out.mkdir()
    (out / "approved-plan.md").write_text("Invented engineering test only; no research training.\n")
    bind(source, out, fixture=True)
    return source, out


def test_archive_lifecycle_and_audits(invented, monkeypatch):
    from super_deli.models import tabular_neural

    def forbidden(*args, **kwargs):
        raise AssertionError("Diagnostic must never train a neural network")

    monkeypatch.setattr(tabular_neural, "_train", forbidden)
    source, out = invented
    config = ViewSignalConfig(fixture=True)
    assert freeze(out, config)["counts"] == {"preflight": 21, "main": 21}
    with pytest.raises(FileNotFoundError):
        run_phase(out, source, "main")
    assert run_phase(out, source, "preflight")["attempted"] == 21
    audit = audit_phase(out, source, "preflight")
    assert audit["native_replays"] == 11 and audit["failed"] == audit["unavailable"] == 0
    assert gate(out, staging_bytes=0)["passed"]
    run_phase(out, source, "main", workers=2)
    audit_phase(out, source, "main")
    summary = report(out)
    assert summary["main_fits"] == 21 and summary["new_neural_training"] == 0
    assert len(read_json(out / "all-spaces.json")) == 21
    with pytest.raises(FileExistsError):
        run_phase(out, source, "main")
    with (out / "approved-plan.md").open("a") as f:
        f.write("changed")
    with pytest.raises(ValueError, match="binding"):
        check_frozen(out)


def test_binding_mutation_and_escape(invented):
    source, out = invented
    binding = read_json(out / "archive-binding.json")
    assert check_binding(source, binding) is None
    (source / "main/invented_n32_0000/extra").write_text("unexpected")
    with pytest.raises(ValueError, match="coverage"):
        check_binding(source, binding)
    with pytest.raises(ValueError, match="escapes"):
        load_context(source, {"path": "../outside", "fixture": True})


def test_numeric_audit_catches_tampering(invented):
    source, out = invented
    entry = read_json(out / "archive-binding.json")["observations"][0]
    context, _ = load_context(source, entry)
    name = "paired_0_standard"
    expected = (context[name + "_labels"], context[name + "_strength"])
    record, data = fit_coordinates(context[name], context, expected=expected)
    audit_fit(context, name, record, data)
    bad = {**data, "core_distances": data["core_distances"] + 1}
    with pytest.raises(AssertionError):
        audit_fit(context, name, record, bad)
    bad_record = copy.deepcopy(record)
    bad_record["geometry"]["scatter_joint"] += 0.1
    with pytest.raises(AssertionError):
        audit_fit(context, name, bad_record, data)
    bad_record = copy.deepcopy(record)
    bad_record["recovery"]["ari"] -= 0.25
    with pytest.raises(AssertionError):
        audit_fit(context, name, bad_record, data)
    with pytest.raises(AssertionError):
        fit_coordinates(context[name], context, expected=(expected[0] + 1, expected[1]))


def test_four_branch_oracle_is_multiclass():
    truth = np.repeat(np.arange(4), 8)
    rows = [(32, 33 + g, 1.0, 8) for g in range(4)]
    rows += [(33 + g, i, 2.0, 1) for g in range(4) for i in range(g * 8, (g + 1) * 8)]
    tree = np.array(
        rows, dtype=[("parent", int), ("child", int), ("value", float), ("cluster_size", int)]
    )
    result = branch_reference(tree, truth)
    assert result["best_jaccard"] == [1.0] * 4 and result["four_disjoint_at_0_8"]
    assert not branch_reference(tree, np.tile(np.arange(4), 8))["four_disjoint_at_0_8"]


def test_scaling_is_invertible_and_ties_withhold():
    rng = np.random.default_rng(901)
    x = rng.normal(size=(32, 4))
    target = x[:, :2] @ np.array([[1, 2], [3, 4]])
    a = linear_recovery(x, target)
    b = linear_recovery(x * [1, 0.001, 100, 0.2], target)
    np.testing.assert_allclose(a["r2"], [1, 1], atol=1e-12)
    np.testing.assert_allclose(a["r2"], b["r2"], atol=1e-12)
    _, ties = neighbors(np.zeros((32, 2)), 5)
    assert ties.all()
    context = {
        "truth": np.repeat(np.arange(4), 8),
        "morphology_standard": np.zeros((32, 4)),
        "latent_morphology": target,
        "latent_ecology": target,
    }
    metrics, _ = geometry(np.zeros((32, 2)), context)
    assert metrics["neighbors_5"]["purity"] is None
    assert metrics["neighbors_15"]["morphology_overlap"] is None


def test_native_spaces_and_fixed_rules(invented):
    source, out = invented
    entry = read_json(out / "archive-binding.json")["observations"][0]
    context, _ = load_context(source, entry)
    assert sum(native_arm(name) is not None for name in REPRESENTATIONS) == 11
    baseline, original = fit_coordinates(context["balanced_traits"], context)
    changed = {**context, "truth": context["truth"][::-1]}
    other, arrays = fit_coordinates(context["balanced_traits"], changed)
    np.testing.assert_array_equal(arrays["labels"], original["labels"])
    assert baseline["parameters"] == other["parameters"]
    with pytest.raises(ValueError, match="coordinates"):
        fit_coordinates(np.full((32, 3), np.nan), context)


def test_gate_requires_measured_budget(invented):
    source, out = invented
    freeze(out, ViewSignalConfig(fixture=True))
    run_phase(out, source, "preflight")
    audit_phase(out, source, "preflight")
    with pytest.raises(ValueError, match="Measured"):
        gate(out, staging_bytes=None)
    assert not gate(out, staging_bytes=1024**3)["passed"]
    with pytest.raises(ValueError, match="gate"):
        run_phase(out, source, "main")
