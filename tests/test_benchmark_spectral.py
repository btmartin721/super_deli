"""Independent graph/eigenspace/QR checks and source/gate integration."""
# ruff: noqa: E402

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from scipy.sparse import csr_matrix
from sklearn.cluster import SpectralClustering

from super_deli.benchmarking.spectral_archive import bind_archive
from super_deli.benchmarking.spectral_audit import audit_graph, audit_phase
from super_deli.benchmarking.spectral_config import SpectralConfig
from super_deli.benchmarking.spectral_fit import build_graph, fit_graph, fit_partition
from super_deli.benchmarking.spectral_fixture import create_fixture
from super_deli.benchmarking.spectral_report import report
from super_deli.benchmarking.spectral_study import freeze, gate, run_phase
from super_deli.io.config import read_json


def test_connected_public_parity_audit_and_corruption():
    x = np.random.default_rng(861).normal(size=(40, 2))
    ids = [f"i{i:03}" for i in range(len(x))]
    r, a = fit_graph(x, ids, 15, max_working_bytes=2**26)
    assert all(p["status"] == "available" and p["method"] == "spectral" for p in r["partitions"])
    audit_graph(x, ids, {"neighbors": 15}, r, a)
    affinity = csr_matrix((a["data"], a["indices"], a["indptr"]), shape=(40, 40))
    for p in r["partitions"]:
        if p["status"] == "available" and p["method"] == "spectral":
            k = p["target_k"]
            labels = SpectralClustering(
                k,
                affinity="precomputed",
                assign_labels="cluster_qr",
                eigen_solver="arpack",
                eigen_tol=0.0,
                random_state=71323,
            ).fit_predict(affinity)
            np.testing.assert_array_equal(a[f"K{k}_labels"], labels)
    bad = {**a, "degrees": a["degrees"] + 1}
    with pytest.raises(AssertionError):
        audit_graph(x, ids, {"neighbors": 15}, r, bad)
    bad = {**a, "K2_embedding": a["K2_embedding"].copy()}
    bad["K2_embedding"][0, 0] += 0.01
    with pytest.raises(AssertionError):
        audit_graph(x, ids, {"neighbors": 15}, r, bad)
    bad = {**a, "eigenvalues": a["eigenvalues"] + 0.01}
    with pytest.raises(AssertionError):
        audit_graph(x, ids, {"neighbors": 15}, r, bad)


def test_components_cutoff_and_failure_retention(monkeypatch):
    rng = np.random.default_rng(862)
    x = np.r_[rng.normal(-20, 0.01, (12, 2)), rng.normal(20, 0.01, (12, 2))]
    ids = [f"i{i:03}" for i in range(len(x))]
    r, a = fit_graph(x, ids, 5, max_working_bytes=2**26)
    assert r["components"] == 2
    assert r["partitions"][0]["method"] == "components"
    assert r["partitions"][1]["reason"] == "unsupported_component_count"
    audit_graph(x, ids, {"neighbors": 5}, r, a)
    x = np.random.default_rng(863).normal(size=(32, 2))
    affinity, data, info = build_graph(x, [str(i) for i in range(32)], 15, max_working_bytes=2**26)
    assert info["components"] == 1
    broken = {**data, "eigenvalues": np.zeros(32)}
    rec, _ = fit_partition(affinity, broken, info, 2)
    assert rec["reason"] == "unresolved_cutoff_eigenspace"
    from super_deli.benchmarking import spectral_fit as m

    def fail(*a, **kw):
        raise ValueError("forced eigensolver failure")

    monkeypatch.setattr(m, "spectral_embedding", fail)
    rec, _ = fit_partition(affinity, data, info, 2)
    assert rec["status"] == "failed" and rec["solver_called"] and "forced" in rec["error"]


def test_exact_ties_identity_and_workspace():
    x = np.array([(i, j) for i in range(5) for j in range(5)], dtype=float)
    ids = [f"i{i:03}" for i in range(25)]
    r, a = fit_graph(x, ids, 5, max_working_bytes=2**26)
    assert r["cutoff_tie_rows"] > 0
    audit_graph(x, ids, {"neighbors": 5}, r, a)
    with pytest.raises(ValueError, match="Tie-row"):
        audit_graph(x, ids, {"neighbors": 5}, {**r, "cutoff_tie_rows": 999}, a)
    for raw, names, budget in (
        (x, ["same"] * 25, 2**26),
        (x, ids, 1),
        (np.full((25, 2), np.nan), ids, 2**26),
    ):
        with pytest.raises(ValueError):
            fit_graph(raw, names, 5, max_working_bytes=budget)


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return create_fixture(tmp_path_factory.mktemp("spectral-source") / "archive")


def test_complete_fixture_protocol(tmp_path, source):
    root = tmp_path / "study"
    root.mkdir()
    (root / "approved-plan.md").write_text(
        "Invented untrained engineering fixture; no species claim."
    )
    bound = bind_archive(source, root, fixture=True)
    assert len(bound["observations"]) == 6
    with pytest.raises(ValueError, match="kind"):
        freeze(root, SpectralConfig())
    f = freeze(root, SpectralConfig(fixture=True))
    assert f["counts"] == {"preflight": 504, "main": 848}
    with pytest.raises(FileNotFoundError):
        run_phase(root, "main")
    assert run_phase(root, "preflight")["evaluations"] == 504
    assert audit_phase(root, "preflight")["passed"]
    assert gate(root, 0)["passed"]
    run_phase(root, "main", 2)
    result = audit_phase(root, "main")
    assert result["evaluations"] == 848 and result["neural_training_fits"] == 0
    assert report(root)["fixture"]
    assert read_json(root / "aggregate.json")["selected_k"] is None
    with pytest.raises(FileExistsError):
        run_phase(root, "main")


def test_failed_storage_gate(tmp_path, source):
    root = tmp_path / "study"
    root.mkdir()
    (root / "approved-plan.md").write_text("Invented storage failure test.")
    bind_archive(source, root, fixture=True)
    freeze(root, SpectralConfig(fixture=True))
    run_phase(root, "preflight")
    audit_phase(root, "preflight")
    assert not gate(root, 2 * 1024**3)["passed"]
    with pytest.raises(ValueError, match="gate"):
        run_phase(root, "main")


def test_numeric_assignment_ties_are_withheld(monkeypatch):
    from super_deli.benchmarking import spectral_fit as module

    x = np.random.default_rng(864).normal(size=(40, 2))
    affinity, data, info = build_graph(x, [str(i) for i in range(40)], 15, max_working_bytes=2**26)
    original = module.rotation

    def tied(maps):
        rotated, piv, s = original(maps)
        rotated[:] = 1
        return rotated, piv, s

    monkeypatch.setattr(module, "rotation", tied)
    record, evidence = fit_partition(affinity, data, info, 2)
    assert record["status"] == "withheld" and record["reason"] == "ambiguous_qr_assignment"
    assert "proposed_labels" in evidence and "labels" not in evidence


def test_frozen_inputs_reject_corruption_and_path_escape(tmp_path, source):
    from super_deli.benchmarking.spectral_archive import load_coordinates
    from super_deli.benchmarking.spectral_study import check_frozen

    root = tmp_path / "study"
    root.mkdir()
    (root / "approved-plan.md").write_text("Invented integrity fixture.")
    b = bind_archive(source, root, fixture=True)
    entry = b["observations"][0]
    with pytest.raises(ValueError, match="path escape"):
        load_coordinates(root, {**entry, "coordinate_file": "../outside.npz"})
    freeze(root, SpectralConfig(fixture=True))
    p = root / entry["coordinate_file"]
    p.write_bytes(p.read_bytes() + b"damaged")
    with pytest.raises(ValueError, match="hash"):
        load_coordinates(root, entry)
    (root / "approved-plan.md").write_text("Changed after freeze")
    with pytest.raises(ValueError, match="binding/plan"):
        check_frozen(root)
