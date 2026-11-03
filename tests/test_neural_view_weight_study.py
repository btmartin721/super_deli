"""Independent-law, numerical guard and fit-freeze lifecycle engineering checks."""

# ruff: noqa: E402
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from super_deli.benchmarking.view_weight_config import ViewWeightStudy, tasks
from super_deli.benchmarking.view_weight_data import generate, reference
from super_deli.io.config import read_json


@pytest.mark.parametrize(
    "case",
    [
        "morphology_only",
        "ecology_only",
        "four_complementary",
        "four_redundant",
        "gaussian_independent",
        "gaussian_correlated",
    ],
)
def test_independent_raw_laws(case):
    t = next(t for t in tasks(ViewWeightStudy(), "preflight") if t["case"] == case)
    a, b = generate(t), reference(t)
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_allclose(a[key], b[key], atol=1e-11)
    assert a["morphology"].shape == (64, 4) and a["ecology"].shape == (64, 8)


def test_live_audit_failure_is_not_swallowed(tmp_path, monkeypatch):
    from super_deli.benchmarking import matched_steps
    from super_deli.benchmarking.view_weight_observer import NumericalAuditFailure, retain_training
    from super_deli.models import tabular_neural

    def reject(*args):
        raise ValueError("injected numerical discrepancy")

    monkeypatch.setattr(matched_steps, "compare", reject)
    guard = {}
    with retain_training(tmp_path, paired=False, guard_state=guard):
        with pytest.raises(NumericalAuditFailure, match="injected"):
            tabular_neural._train(np.random.default_rng(1).normal(size=(16, 4)), 0)
    assert guard["failed"]
    events = read_json(tmp_path / "optimizer-checks.json")
    assert events[0]["status"] == "failed" and events[0]["numerical_guard"]


def test_fixture_lifecycle(tmp_path):
    from super_deli.benchmarking.view_weight_archive import bind, create_fixture
    from super_deli.benchmarking.view_weight_report import report
    from super_deli.benchmarking.view_weight_study import (
        audit_phase,
        check_frozen,
        freeze,
        gate,
        run_phase,
    )

    source = create_fixture(tmp_path / "source")
    root = tmp_path / "study"
    root.mkdir()
    (root / "approved-plan.md").write_text("Invented engineering lifecycle only.\n")
    bind(source, root, fixture=True)
    config = ViewWeightStudy(fixture=True)
    freeze(root, config)
    assert run_phase(root, source, "preflight", workers=1)["attempted"] == 25
    assert audit_phase(root, source, "preflight")["verified_optimizer_updates"] == 24 * 500
    assert gate(root, staging_bytes=0)["passed"]
    assert run_phase(root, source, "main", workers=1)["attempted"] == 26
    assert audit_phase(root, source, "main")["verified_optimizer_updates"] == 25 * 500
    assert report(root)["failed_points"] == 0
    (root / "approved-plan.md").write_text("tampered")
    with pytest.raises(ValueError, match="Frozen"):
        check_frozen(root)
