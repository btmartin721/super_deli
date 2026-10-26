"""Geometry, explicit branch-selection cases, parity and source replay checks."""

# ruff: noqa: E402
from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from sklearn.cluster._hdbscan._tree import HIERARCHY_dtype, _condense_tree, tree_to_labels

from super_deli.benchmarking.moons_archive import simulate_latents
from super_deli.benchmarking.moons_audit import (
    _same_condensation,
    audit_arm,
    independent_condensation,
)
from super_deli.benchmarking.moons_fit import fit_diagnostic_arm
from super_deli.benchmarking.moons_geometry import graph_diagnostics
from super_deli.benchmarking.moons_hierarchy import selection_trace, validate_linkage
from super_deli.core.data import Feature
from super_deli.core.tabular import TabularTable
from super_deli.models.hdbscan_config import HDBSCANConfig
from super_deli.models.tabular_hdbscan import fit_tabular_hdbscan

pytestmark = pytest.mark.baseline


def table(x):
    return TabularTable(
        "traits",
        "morphological",
        "specimen",
        (Feature("trait_1", "continuous"), Feature("trait_2", "continuous")),
        tuple(f"unit_{i:04d}" for i in range(len(x))),
        tuple(f"unit_{i:04d}" for i in range(len(x))),
        tuple(tuple(float(v) for v in row) for row in x),
    )


def known_tree():
    return np.array(
        [
            (0, 1, 0.1, 2),
            (2, 3, 0.12, 2),
            (4, 5, 0.15, 2),
            (6, 7, 0.18, 2),
            (8, 9, 0.7, 4),
            (10, 11, 0.8, 4),
            (12, 13, 1.4, 8),
        ],
        dtype=HIERARCHY_dtype,
    )


@pytest.mark.parametrize(
    "method,single,k", [("eom", True, 1), ("eom", False, 2), ("leaf", True, 2)]
)
def test_known_parent_mass_can_select_one_despite_two_surviving_branches(method, single, k):
    tree = known_tree()
    c = _condense_tree(tree, 3)
    _same_condensation(independent_condensation(tree, 8, 3), c, 8)
    trace, labels, strength = selection_trace(c, method=method, allow_single_cluster=single)
    expected, prob = tree_to_labels(
        tree, min_cluster_size=3, cluster_selection_method=method, allow_single_cluster=single
    )
    np.testing.assert_array_equal(labels, expected)
    np.testing.assert_allclose(strength, prob)
    assert len(set(labels) - {-1}) == k
    assert trace["root_margin"] > 0 and trace["surviving_leaves"] == 2


@pytest.mark.parametrize("minimum", [2, 3, 5, 10])
def test_condensation_leaf_mass_is_conserved(minimum):
    tree = known_tree()
    c = _condense_tree(tree, minimum)
    _same_condensation(independent_condensation(tree, 8, minimum), c, 8)
    assert sorted(c[c["cluster_size"] == 1]["child"]) == list(range(8))


@pytest.mark.parametrize(
    "arm",
    [
        "baseline",
        "no_single",
        "leaf",
        "samples_2",
        "samples_10",
        "cluster_3",
        "cluster_10",
        "center_only",
    ],
)
def test_all_arms_and_independent_density_audits(arm):
    data = simulate_latents("moons", 64, [7129])
    m, a = fit_diagnostic_arm(table(data["x"]), arm)
    assert m["hierarchy_status"] == "available", m.get("hierarchy_error")
    graph, core, edges = graph_diagnostics(
        a["scores"], data["truth"], m["parameters"]["min_samples"]
    )
    a.update(core_distances=core, mst_edges=edges)
    m["graph"] = graph
    assert audit_arm(data["x"], data["truth"], m, a)["hierarchy_status"] == "available"
    if arm == "baseline":
        fit = fit_tabular_hdbscan(table(data["x"]), config=HDBSCANConfig(True))
        np.testing.assert_array_equal(a["labels"], fit.point.labels)
        np.testing.assert_allclose(a["scores"], fit.point.scores)
        np.testing.assert_allclose(a["strength"], fit.point.strength)


def test_latent_reconstruction_matches_locked_source_generator():
    from super_deli.benchmarking.density_study import generate_raw

    x, y = generate_raw("moons", 64, [93709, 94217, 3, 64, 0])
    d = simulate_latents("moons", 64, [93709, 94217, 3, 64, 0])
    np.testing.assert_array_equal(x, d["x"])
    np.testing.assert_array_equal(y, d["truth"])
    np.testing.assert_array_equal(d["x"], d["latent"] + d["noise"])


def test_mutual_reachability_threshold_and_self_neighbor_convention():
    x = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [10.0, 10.0], [10.1, 10.0], [10.0, 10.1]])
    truth = np.repeat([0, 1], 3)
    graph, core, edges = graph_diagnostics(x, truth, 2)
    np.testing.assert_allclose(core, 0.1, atol=1e-12)
    assert graph["mutual_threshold_separable"] and graph["cross_neighbor_fraction"] == 0
    assert len(edges) == 5


def test_invalid_hierarchy_is_not_interpreted():
    tree = known_tree()
    bad = tree.copy()
    bad[0]["left_node"] = 500
    with pytest.raises(ValueError):
        validate_linkage(bad, 8)
    bad = tree.copy()
    bad[0]["value"] = 0
    with pytest.raises(ValueError):
        validate_linkage(bad, 8)


def test_scope_preserves_full_dimension_and_complete_features():
    x = simulate_latents("moons", 64, 81)["x"]
    with pytest.raises(ValueError):
        fit_diagnostic_arm(
            replace(table(x), values=((None, 1.0),) + table(x).values[1:]), "baseline"
        )


@pytest.fixture(scope="module")
def invented_archive(tmp_path_factory):
    from super_deli.benchmarking.moons_fixture import build_fixture

    root = tmp_path_factory.mktemp("moons-invented") / "archive"
    assert build_fixture(root)["synthetic_fixture"]
    return root


def test_immutable_source_audit_and_exact_small_workflow(invented_archive, tmp_path):
    from super_deli.benchmarking.moons_archive import audit_archive
    from super_deli.benchmarking.moons_audit import audit_moons_diagnostic
    from super_deli.benchmarking.moons_config import MoonsDiagnosticConfig
    from super_deli.benchmarking.moons_study import moons_storage_gate, run_moons_diagnostic
    from super_deli.core.provenance import fingerprint_file

    original = {
        p.relative_to(invented_archive).as_posix(): fingerprint_file(p).sha256
        for p in invented_archive.rglob("*")
        if p.is_file()
    }
    c = MoonsDiagnosticConfig(datasets_per_cell=1, sample_sizes=(32,), min_free_bytes=1)
    a = tmp_path / "archive-audit"
    summary = audit_archive(invented_archive, output_dir=a, sample_sizes=c.sample_sizes)
    assert summary["observations"] == 5 and summary["retained_draws"] == 5
    pref = tmp_path / "preflight"
    run = run_moons_diagnostic(
        c,
        archive_root=invented_archive,
        phase="preflight",
        output_dir=pref,
        archive_audit=a / "summary.json",
    )
    assert run["attempted_fits"] == 25 and run["successful_fits"] == 25
    audit = tmp_path / "preflight-audit"
    result = audit_moons_diagnostic(pref, archive_root=invented_archive, output_dir=audit)
    assert result["audited_successful_fits"] == 25 and len(result["primary_rates"]) == 24
    gate = tmp_path / "gate.json"
    assert (
        moons_storage_gate(
            c,
            probe_dir=pref,
            audit_file=audit / "summary.json",
            artifact_root=tmp_path,
            output_file=gate,
        )["status"]
        == "passed"
    )
    main = tmp_path / "main"
    result = run_moons_diagnostic(
        c,
        archive_root=invented_archive,
        phase="main",
        output_dir=main,
        archive_audit=a / "summary.json",
        gate=gate,
    )
    assert result["attempted_fits"] == 69
    assert (
        audit_moons_diagnostic(
            main, archive_root=invented_archive, output_dir=tmp_path / "main-audit"
        )["verified_replays"]
        == 10
    )
    assert original == {
        p.relative_to(invented_archive).as_posix(): fingerprint_file(p).sha256
        for p in invented_archive.rglob("*")
        if p.is_file()
    }
    with pytest.raises(FileExistsError):
        run_moons_diagnostic(
            c,
            archive_root=invented_archive,
            phase="preflight",
            output_dir=pref,
            archive_audit=a / "summary.json",
        )


def test_failed_setup_is_retained_without_model_fallback(tmp_path):
    from super_deli.benchmarking.moons_study import safe_run_observation

    row = {
        "key": "broken",
        "origin": "archived",
        "case": "moons_008",
        "n": 64,
        "dataset": 0,
        "seed": [91],
        "source_key": "absent",
    }
    r = safe_run_observation((str(tmp_path), str(tmp_path / "missing"), row))
    assert r["setup_status"] == "failed" and not r["arms"]
    assert (tmp_path / "broken/checksums.json").is_file()


def test_unsupported_library_version_is_reported(monkeypatch):
    from super_deli.benchmarking import moons_hierarchy

    monkeypatch.setattr(moons_hierarchy, "version", lambda _: "2.0.0")
    with pytest.raises(ValueError, match="version"):
        moons_hierarchy.retained_hierarchy(object(), 8, 3)


def test_independent_audit_detects_altered_graph():
    d = simulate_latents("moons", 64, 87)
    m, a = fit_diagnostic_arm(table(d["x"]), "baseline")
    graph, core, edges = graph_diagnostics(a["scores"], d["truth"], 5)
    a.update(core_distances=core + 1, mst_edges=edges)
    m["graph"] = graph
    with pytest.raises(AssertionError):
        audit_arm(d["x"], d["truth"], m, a)
