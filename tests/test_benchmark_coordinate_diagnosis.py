"""Invented frozen fixtures, independent numerical checks and actual failure guards."""

# ruff: noqa: E402
import os
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from super_deli.benchmarking.coordinate_archive import bind, load_context
from super_deli.benchmarking.coordinate_audit import audit_mixture, gaussian_reference
from super_deli.benchmarking.coordinate_config import (
    CoordinateStudy,
    tasks,
)
from super_deli.benchmarking.coordinate_fit import density, mixture
from super_deli.benchmarking.coordinate_fixture import create_fixture
from super_deli.benchmarking.coordinate_report import report
from super_deli.benchmarking.coordinate_study import (
    audit_phase,
    check_frozen,
    freeze,
    gate,
    run_phase,
)
from super_deli.io.config import read_json


def test_invented_full_lifecycle(tmp_path):
    if os.environ.get("SUPER_DELI_COORDINATE_TEST_CHILD") != "1":
        env = os.environ.copy()
        env["SUPER_DELI_COORDINATE_TEST_CHILD"] = "1"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import runpy,sys;from pathlib import Path;"
                "runpy.run_path(sys.argv[1])['test_invented_full_lifecycle'](Path(sys.argv[2]))",
                __file__,
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return
    source = create_fixture(tmp_path / "source")
    root = tmp_path / "study"
    root.mkdir()
    (root / "approved-plan.md").write_text("Invented engineering fixture only.\n")
    bind(source, root, fixture=True)
    freeze(root, CoordinateStudy(fixture=True))
    pre = run_phase(root, source, "preflight", workers=1)
    assert pre["completed"] and pre["attempted"] == 30 and pre["neural_fits"] == 0
    audit = audit_phase(root, source, "preflight")
    assert audit["passed"] and audit["verified_optimizer_updates"] == 0
    assert gate(root, staging_bytes=0)["passed"]
    main = run_phase(root, source, "main", workers=1)
    assert main["completed"] and main["attempted"] == 30
    assert audit_phase(root, source, "main")["passed"]
    result = report(root, source)
    assert (
        result["observations"] == 1
        and result["neural_training"] == 0
        and result["new_datasets"] == 0
    )
    assert "records" not in result["audits"]["main"]
    (root / "approved-plan.md").write_text("changed")
    with pytest.raises(ValueError, match="Frozen"):
        check_frozen(root)


@pytest.mark.parametrize("family", ["full", "tied"])
def test_independent_gaussian_density_and_bic(family):
    rng = np.random.default_rng(55001)
    scores = np.concatenate([rng.normal(loc, 0.5, (16, 4)) for loc in (-4.0, 4.0)])
    record, a = mixture(scores, family)
    record["clusterer"] = "gmm_full" if family == "full" else "gmm_shared"
    assert record["status"] == "available"
    verified, candidates = audit_mixture(scores, record, a)
    assert verified == 4 and candidates[2]["eligible"]
    modified = {**a, "k2_weights": a["k2_weights"] * 2}
    with pytest.raises(AssertionError):
        gaussian_reference(scores, modified, 2, family)


def test_partial_convergence_failure_withholds_search(monkeypatch):
    from sklearn.mixture import GaussianMixture

    original = GaussianMixture.fit
    called = []

    def failed(estimator, x, *args, **kwargs):
        called.append(estimator.n_components)
        if estimator.n_components == 3:
            raise ValueError("injected candidate failure")
        return original(estimator, x, *args, **kwargs)

    monkeypatch.setattr(GaussianMixture, "fit", failed)
    z = np.random.default_rng(31).normal(size=(32, 4))
    row, a = mixture(z, "full")
    row["clusterer"] = "gmm_full"
    assert row["status"] == "failed" and row["selected_k"] is None and called == [1, 2, 3]
    assert "labels" not in a and "k2_membership" in a
    assert audit_mixture(z, row, a)[0] == 2
    assert GaussianMixture.fit is failed


def test_bound_source_and_replay_reject_changes(tmp_path):
    from super_deli.benchmarking.coordinate_fit import AuditFailure
    from super_deli.benchmarking.view_weight_archive import check_binding

    source = create_fixture(tmp_path / "source")
    out = tmp_path / "binding"
    bind(source, out, fixture=True)
    context = load_context(source, tasks(CoordinateStudy(fixture=True), "main")[0])
    point = context["spaces"]["separate_0"]
    with pytest.raises(AuditFailure):
        density({**point, "labels": np.zeros(len(point["labels"]), dtype=int) + 40}, "separate_0")
    (source / "fixture.json").write_text("tampered")
    with pytest.raises(ValueError, match="source changed"):
        check_binding(source, read_json(out / "replay-binding.json"))
