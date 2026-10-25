"""Whole-locality multiplicities, leakage, projections and failure-aware stability."""

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from super_deli.benchmarking.predictive_audit import independent_simulation  # noqa: E402
from super_deli.benchmarking.predictive_bootstrap_audit import audit_draw, audit_stability  # noqa: E402
from super_deli.io.config import read_json  # noqa: E402
from super_deli.models.grouped_null import write_raw_views  # noqa: E402
from super_deli.models.predictive_bootstrap_config import PredictiveBootstrapConfig  # noqa: E402
from super_deli.models.predictive_bootstrap_study import PredictiveBootstrapStudy  # noqa: E402
from super_deli.models.predictive_stability import summarize_partitions  # noqa: E402
from super_deli.workflows.predictive_bootstrap import run_predictive_bootstrap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    c = PredictiveBootstrapStudy.from_dict(
        read_json(ROOT / "examples/predictive-stability/smoke.json")
    )
    cell = c.cells[4]
    v, layout, t = independent_simulation(cell, cell.n_units, 202612140001)
    base = tmp_path_factory.mktemp("predictive-locality-bootstrap")
    dataset = write_raw_views(v, layout, base / "inputs")
    bc = PredictiveBootstrapConfig(2, 202612140002)
    result = run_predictive_bootstrap(
        dataset,
        model_config=c.model,
        screen_config=c.screening,
        bootstrap_config=bc,
        base_dir=base / "inputs",
        output_dir=base / "workflow",
    )
    return base, c, v, layout, result


def test_full_draw_pipeline_and_independent_audit(fitted):
    base, c, v, layout, r = fitted
    assert r["successful_fits"] == 18 and not r["failed_fits"] and not r["unattempted_fits"]
    assert r["stability"]["successful"] == 2
    assignments = read_json(base / "workflow/point/fold_assignments.json")
    for row in r["replicates"]:
        draw = np.asarray(row["training_indices"])
        assert len(set(draw)) < len(draw)
        assert (
            audit_draw(
                base / "workflow" / row["prefix"], v, layout, draw, c.model, assignments, row
            )
            == 6
        )
    audit_stability(base / "workflow", r)


def test_expanded_weights_and_distinct_locality_constraints(fitted):
    base, _, v, layout, r = fitted
    row = r["replicates"][0]
    draw = np.asarray(row["training_indices"])
    path = base / "workflow" / row["prefix"] / "full/run"
    targets = np.asarray(layout["views"][0]["target_ids"])
    means = np.stack([v["morphological"][targets == u].mean(0) for u in layout["unit_ids"]])
    weights = np.bincount(draw, minlength=len(means))
    mean = np.average(means, axis=0, weights=weights)
    scale = np.sqrt(np.average((means - mean) ** 2, axis=0, weights=weights))
    with np.load(path / "view_000/parameters/pca.npz") as p:
        np.testing.assert_allclose(p["feature_means"], mean, atol=1e-13)
        np.testing.assert_allclose(p["feature_scales"], scale, atol=1e-13)
    with np.load(path / "gmm_candidates.npz") as p:
        for candidate in read_json(path / "candidates.json"):
            if candidate["status"] == "insufficient_samples":
                continue
            k = candidate["k"]
            labels = p[f"k{k}_labels"]
            distinct = [len(set(draw[labels == j])) for j in range(k)]
            assert candidate["distinct_unit_sizes"] == distinct
            assert candidate["hard_cluster_sizes"] == np.bincount(labels, minlength=k).tolist()


def record(labels, k, draw=None, status="complete"):
    return dict(
        labels=labels,
        selected_k=k,
        training_indices=[0, 0, 0, 0] if draw is None else draw,
        status=status,
    )


def test_labels_merging_failure_bounds_and_sparse_oob():
    rows = [record([1, 1, 0, 0], 2), record([0, 0, 0, 0], 1), record([], 1, status="failed")]
    summary, a = summarize_partitions(np.array([0, 0, 1, 1]), 2, rows)
    assert summary["status"] == "unavailable" and summary["successful"] == 2
    np.testing.assert_allclose(a["exact_cluster_counts"], [1, 1])
    assert np.isnan(a["all_frequency"][0, 2])
    assert a["all_conditional_frequency"][0, 2] == 0.5
    assert a["all_failure_lower"][0, 2] == pytest.approx(1 / 3)
    assert a["all_failure_upper"][0, 2] == pytest.approx(2 / 3)
    assert np.isnan(a["oob_frequency"]).all()
    assert summary["clusters"][0]["all_cross_merging_conditional"] == 0.5
    assert summary["clusters"][0]["all_within_cohesion_conditional"] == 1
    with pytest.raises(ValueError):
        summarize_partitions(np.array([0, 0, 1, 1]), 2, [record([0, 1], 2)])


def test_oob_threshold_and_exact_recovery():
    rows = [record([0, 0, 1, 1], 2) for _ in range(10)]
    s, a = summarize_partitions(np.array([0, 0, 1, 1]), 2, rows)
    assert a["oob_successful"][1, 2] == 10 and a["oob_frequency"][1, 2] == 0
    assert np.isnan(a["oob_frequency"][0, 2])
    assert s["clusters"][0]["exact_recovery_count"] == 10
    assert a["selected_k_frequency"][1] == 1
    assert a["selected_k_conditional_mc_low"][1] < 1


def test_empty_fold_is_retained_and_withholds_without_replacement(fitted, tmp_path):
    from super_deli.workflows.predictive_resampling import occurrence_screen

    base, c, v, layout, _ = fitted
    assignments = read_json(base / "workflow/point/fold_assignments.json")
    excluded = set(assignments[0]["validation_indices"])
    eligible = [i for i in range(len(layout["unit_ids"])) if i not in excluded]
    draw = np.resize(eligible, len(layout["unit_ids"]))
    r = occurrence_screen(tmp_path / "empty", v, layout, draw, c.model, assignments)
    assert r["selection"]["status"] == "withheld"
    assert r["failed_fits"] >= 1
    assert read_json(tmp_path / "empty/fold_000/record.json")["error"].startswith("Empty")
    assert not (tmp_path / "empty/original_predictions.npz").exists()


def test_bootstrap_fit_failure_has_no_fake_partition(fitted, tmp_path, monkeypatch):
    import super_deli.models.integrative as integration
    from super_deli.workflows.predictive_resampling import occurrence_screen

    base, c, v, layout, _ = fitted

    def fail(*args, **kwargs):
        raise ValueError("deliberate bootstrap failure")

    monkeypatch.setattr(integration, "_fit_once", fail)
    r = occurrence_screen(
        tmp_path / "failed",
        v,
        layout,
        np.arange(len(layout["unit_ids"])),
        c.model,
        read_json(base / "workflow/point/fold_assignments.json"),
    )
    assert r["attempted_fits"] == r["failed_fits"] == 1 and r["unattempted_fits"] == 5
    assert r["selection"]["screened_k"] is None
    assert not (tmp_path / "failed/original_predictions.npz").exists()


def test_training_map_excludes_heldout_raw_values(fitted, tmp_path):
    from super_deli.workflows.predictive_resampling import occurrence_search

    base, c, v, layout, r = fitted
    original = r["replicates"][0]
    indices = np.asarray(original["training_indices"])
    missing = set(range(len(layout["unit_ids"]))) - set(indices)
    changed = {k: array.copy() for k, array in v.items()}
    for view in layout["views"]:
        rows = [
            i for i, u in enumerate(view["target_ids"]) if layout["unit_ids"].index(u) in missing
        ]
        changed[view["kind"]][rows] += 10000
    occurrence_search(tmp_path, "changed", changed, layout, indices, c.model)
    old = base / "workflow" / original["prefix"] / "full/run"
    for name in [
        "fusion.npz",
        "gmm_candidates.npz",
        "view_000/parameters/pca.npz",
        "view_001/parameters/pca.npz",
    ]:
        with np.load(old / name) as a, np.load(tmp_path / "changed/run" / name) as b:
            assert set(a.files) == set(b.files)
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
