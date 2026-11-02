"""Source-free point-model branch tests and independent numeric corruption checks."""
# ruff: noqa: E402

from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from sklearn.cluster import SpectralClustering

from super_deli.benchmarking.unknown_k_graph_audit import (
    audit_embedding,
    audit_geometry,
    audit_spectral,
)
from super_deli.core.data import Feature
from super_deli.core.tabular import TabularTable
from super_deli.models import spectral_graph as graph_model
from super_deli.models.spectral_config import SpectralConfig
from super_deli.models.tabular_spectral import fit_tabular_spectral, prepare


def table(x):
    return TabularTable(
        "traits",
        "morphological",
        "specimen",
        tuple(Feature(f"v{i}", "continuous") for i in range(x.shape[1])),
        tuple(f"s{i:03d}" for i in range(len(x))),
        tuple(f"s{i:03d}" for i in range(len(x))),
        tuple(map(tuple, x.tolist())),
    )


def config(p=2, neighbors=15):
    return SpectralConfig(True, tuple(f"v{i}" for i in range(p)), neighbors)


@pytest.mark.parametrize(
    "bad",
    [
        "missing",
        "nan",
        "bool",
        "constant",
        "duplicate",
        "count",
        "covariate",
        "level",
        "small",
        "workspace",
    ],
)
def test_input_guards(bad):
    x = np.random.default_rng(99051).normal(size=(40, 2))
    t = table(x)
    c = config()
    if bad in {"missing", "nan", "bool", "constant"}:
        x = x.astype(object)
        if bad == "constant":
            x[:, 0] = 1.0
        else:
            x[0, 0] = {"missing": None, "nan": float("nan"), "bool": True}[bad]
        t = table(x)
    if bad == "duplicate":
        t = replace(t, unit_ids=("same",) * 40)
    if bad == "count":
        t = replace(t, features=(Feature("v0", "count"), t.features[1]))
    if bad == "covariate":
        t = replace(t, features=(Feature("v0", "continuous", role="covariate"), t.features[1]))
    if bad == "level":
        t = replace(t, level="unknown")
    if bad == "small":
        t = table(x[:24])
    if bad == "workspace":
        c = replace(c, max_working_bytes=1)
    with pytest.raises((ValueError, FloatingPointError)):
        prepare(t, c)


def test_gap_rule_boundaries_and_ties():
    _, r = graph_model.propose(np.array([0, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]), 6)
    assert r["proposed_k"] == 1 and r["reason"] is None
    _, r = graph_model.propose(np.arange(7, dtype=float), 6)
    assert r["reason"] == "unresolved_gap_choice"
    _, r = graph_model.propose(np.array([0, 0.01, 0.02, 0.03, 0.04, 0.05, 1.0]), 6)
    assert r["proposed_k"] == 6 and r["reason"] == "search_boundary"


def test_independent_graph_selection_and_corruptions():
    t = table(np.random.default_rng(99052).normal(size=(48, 8)))
    c = config(8)
    record, a = fit_tabular_spectral(t, config=c)
    x = np.array(t.values)
    audit_spectral(x, t.unit_ids, c, record, a)
    for key in ("degrees", "eigenvalues", "gaps", "means", "scales"):
        bad = {**a, key: a[key] + 0.02}
        with pytest.raises((AssertionError, ValueError)):
            audit_spectral(x, t.unit_ids, c, record, bad)
    with pytest.raises(ValueError, match="Outcome-free"):
        audit_spectral(x, t.unit_ids, c, {**record, "proposed_k": 99}, a)
    with pytest.raises(ValueError, match="Tie-row"):
        audit_spectral(x, t.unit_ids, c, {**record, "cutoff_tie_rows": 999}, a)


def test_public_embedding_parity_for_all_timing_dimensions():
    x = np.random.default_rng(99053).normal(size=(48, 8))
    x = (x - x.mean(0)) / x.std(0)
    ids = [str(i) for i in range(48)]
    c = config(8)
    affinity, a, r = graph_model.graph(x, ids, c)
    lap, values, vectors, _ = audit_geometry(x, ids, c, r, a)
    for target in range(2, 7):
        trace = {**a}
        evidence, diagnostic = graph_model.embedding(affinity, trace, target, 71323)
        trace.update(evidence)
        assert (
            audit_embedding(trace, diagnostic, lap, values, vectors, target) == diagnostic["reason"]
        )
        public = SpectralClustering(
            target,
            affinity="precomputed",
            assign_labels="cluster_qr",
            eigen_solver="arpack",
            eigen_tol=0.0,
            random_state=71323,
        ).fit_predict(affinity)
        np.testing.assert_array_equal(public, trace["proposed_labels"])
        bad = {**trace, "embedding": trace["embedding"].copy()}
        bad["embedding"][0, 0] += 0.01
        with pytest.raises((AssertionError, ValueError)):
            audit_embedding(bad, diagnostic, lap, values, vectors, target)
        if target == 2:
            for key in ("rotated", "assignment_margin", "qr_singular_values"):
                bad = {**trace, key: trace[key] + 0.01}
                with pytest.raises((AssertionError, ValueError)):
                    audit_embedding(bad, diagnostic, lap, values, vectors, target)


def test_components_and_overfragmentation():
    rng = np.random.default_rng(99054)
    centers = np.array([[-100, -100], [-100, 100], [100, -100], [100, 100]])
    t = table(np.vstack([rng.normal(c, 0.01, (12, 2)) for c in centers]))
    c = config(neighbors=5)
    r, a = fit_tabular_spectral(t, config=c)
    assert r["status"] == "available" and r["selected_k"] == 4 and r["method"] == "components"
    audit_spectral(np.array(t.values), t.unit_ids, c, r, a)
    x = np.vstack([rng.normal([100 * j, 10 * j * j], 0.001, (8, 2)) for j in range(8)])
    r, a = fit_tabular_spectral(table(x), config=c)
    assert r["status"] == "withheld" and r["selected_k"] is None and "labels" not in a


def force_k2(monkeypatch):
    original = graph_model.propose

    def choose(values, max_k):
        gaps, _ = original(values, max_k)
        return gaps, {"proposed_k": 2, "gap_lead": 1.0, "reason": None}

    monkeypatch.setattr(graph_model, "propose", choose)


def test_failure_and_small_cluster_never_fall_back(monkeypatch):
    t = table(np.random.default_rng(99055).normal(size=(40, 2)))
    c = config()
    force_k2(monkeypatch)

    def fail():
        raise ValueError("forced QR failure after embedding")

    monkeypatch.setattr(graph_model, "qr_helper", fail)
    r, a = fit_tabular_spectral(t, config=c)
    assert (
        r["status"] == "failed"
        and r["selected_k"] is None
        and "embedding" in a
        and "labels" not in a
    )
    assert "forced QR" in r["error"]

    def small(affinity, arrays, target, seed):
        return {"proposed_labels": np.r_[0, np.ones(39, dtype=int)]}, {"reason": None}

    monkeypatch.setattr(graph_model, "embedding", small)
    r, a = fit_tabular_spectral(t, config=c)
    assert r["reason"] == "small_cluster" and r["status"] == "withheld" and r["selected_k"] is None
    assert "labels" not in a


def test_ties_are_identity_ordered():
    x = np.array([(i, j) for i in range(6) for j in range(6)], dtype=float)
    c = config(neighbors=5)
    ids = [f"s{i:03d}" for i in range(36)]
    _, a, r = graph_model.graph(x, ids, c)
    assert r["cutoff_tie_rows"] > 0
    audit_geometry(x, ids, c, r, a)
    with pytest.raises(ValueError):
        graph_model.graph(x, ["same"] * 36, c)


def test_withheld_workflow_has_no_assignments(tmp_path, monkeypatch):
    from super_deli.benchmarking.neural_study import write_dataset
    from super_deli.benchmarking.unknown_k_data import model_config
    from super_deli.workflows.tabular_spectral import run_tabular_spectral

    raw = tmp_path / "raw"
    dataset = write_dataset(np.random.default_rng(99056).normal(size=(40, 8)), raw)
    original = graph_model.propose

    def boundary(values, max_k):
        gaps, _ = original(values, max_k)
        return gaps, {"proposed_k": max_k, "gap_lead": 1.0, "reason": "search_boundary"}

    monkeypatch.setattr(graph_model, "propose", boundary)
    out = tmp_path / "withheld"
    result = run_tabular_spectral(
        dataset, model_config=model_config("spectral_15"), base_dir=raw, output_dir=out
    )
    assert result["status"] == "withheld" and result["selected_k"] is None
    assert (
        result["inference"]["p_value"] is None and result["stability"]["status"] == "not_computed"
    )
    assert (out / "point.npz").exists() and not (out / "assignments.csv").exists()
