"""Known groups, null structure, complete-pipeline resampling and failure accounting."""

# ruff: noqa: E402

from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from sklearn.metrics import adjusted_rand_score

from super_deli.models import shape_gmm
from super_deli.models.shape_cluster_config import ShapeClusterConfig
from super_deli.models.shape_gmm import fit_shape_gmm
from super_deli.models.shape_pca import project_shape_pca

pytestmark = pytest.mark.baseline


def simulate(kind="groups", dimension=2, seed=71):
    rng = np.random.default_rng(seed)
    base = np.array([[-1, -0.7], [0, -1], [1, -0.7], [1, 0.7], [0, 1], [-1, 0.7]])
    x = np.broadcast_to(base, (80, 6, 2)).copy()
    truth = np.repeat([0, 1], 40)
    a = (truth * 0.32 - 0.16 if kind == "groups" else 0) + rng.normal(0, 0.015, 80)
    x[:, 4, 1] += a
    x[:, 2:4, 0] += rng.normal(0, 0.02, (80, 1))
    x += rng.normal(0, 0.001, x.shape)
    if dimension == 3:
        x = np.concatenate([x, rng.normal(0, 0.01, (80, 6, 1))], axis=2)
    return x, truth


def settings(**kwargs):
    return replace(ShapeClusterConfig(True, n_bootstrap=5, n_init=3, max_k=3), **kwargs)


def fit(x, config=None):
    return fit_shape_gmm(x, [f"u{i}" for i in range(len(x))], config=config or settings())


@pytest.mark.parametrize("dimension", [2, 3])
def test_known_groups_and_full_covariance(dimension):
    x, truth = simulate(dimension=dimension)
    result = fit(x)
    assert result.point.selected_k == 2
    assert adjusted_rand_score(truth, result.point.labels) == 1
    assert result.point.mixture.covariance_type == "full"
    np.testing.assert_allclose(result.point.membership.sum(axis=1), 1)
    same = np.equal.outer(truth, truth)
    s = result.stability
    assert s.status == "available"
    assert s.arrays["coassignment"][same].mean() > 0.95
    assert s.arrays["coassignment"][~same].mean() < 0.05
    projected = project_shape_pca(x, model=result.shape.model)
    np.testing.assert_allclose(
        result.point.membership,
        result.point.mixture.predict_proba(projected.scores)[:, result.point.component_order],
    )


def test_one_group_is_an_eligible_and_selected_candidate():
    result = fit(simulate("null")[0], settings(n_bootstrap=0))
    assert result.point.selected_k == 1
    assert result.point.candidates[0]["k"] == 1
    assert result.stability.status == "not_computed" and not result.stability.arrays


def test_pose_and_size_invariance_and_seed_reproducibility():
    x, _ = simulate()
    before = x.copy()
    first, second = fit(x), fit(x)
    np.testing.assert_array_equal(x, before)
    np.testing.assert_array_equal(first.point.membership, second.point.membership)
    assert first.stability.replicates == second.stability.replicates
    rng = np.random.default_rng(123)
    angles = rng.uniform(-np.pi, np.pi, len(x))
    rotations = np.array([[[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]] for t in angles])
    moved = x @ rotations * rng.uniform(0.5, 5, (len(x), 1, 1)) + rng.normal(0, 20, (len(x), 1, 2))
    transformed = fit(moved, settings(n_bootstrap=0))
    np.testing.assert_allclose(first.point.membership, transformed.point.membership, atol=1e-7)
    short = fit(x, settings(n_bootstrap=0))
    np.testing.assert_array_equal(short.point.membership, first.point.membership)


def test_every_draw_refits_raw_shapes_then_predicts_original_units(monkeypatch):
    x, _ = simulate()
    original_fit, original_project = shape_gmm.fit_shape_pca, shape_gmm.project_shape_pca
    draws, references = [], []

    def capture_fit(coordinates, **kwargs):
        draws.append(np.array(coordinates))
        return original_fit(coordinates, **kwargs)

    def capture_project(coordinates, **kwargs):
        references.append(kwargs["model"])
        np.testing.assert_array_equal(coordinates, x)
        return original_project(coordinates, **kwargs)

    monkeypatch.setattr(shape_gmm, "fit_shape_pca", capture_fit)
    monkeypatch.setattr(shape_gmm, "project_shape_pca", capture_project)
    result = fit(x)
    assert len(draws) == 6 and len(references) == 5
    assert all(model is not result.shape.model for model in references)
    for draw, record in zip(draws[1:], result.stability.replicates, strict=True):
        np.testing.assert_array_equal(draw, x[record["training_indices"]])
        assert draw.shape == x.shape and record["unique_training_units"] < len(x)


def test_pairwise_oob_denominators_and_exact_recovery_are_auditable():
    x, _ = simulate()
    result = fit(x)
    s, n = result.stability, len(x)
    counts = np.zeros((n, n), int)
    oob_counts, requested = counts.copy(), counts.copy()
    for record in s.replicates:
        labels = np.array(record["labels"])
        same = np.equal.outer(labels, labels)
        counts += same
        oob = np.zeros(n, bool)
        oob[record["oob_indices"]] = True
        pairs = np.logical_and.outer(oob, oob)
        requested += pairs
        oob_counts += same & pairs
    np.testing.assert_array_equal(s.arrays["coassignment_counts"], counts)
    np.testing.assert_array_equal(s.arrays["oob_requested"], requested)
    np.testing.assert_array_equal(s.arrays["oob_coassignment_counts"], oob_counts)
    observed = requested > 0
    np.testing.assert_allclose(
        s.arrays["oob_coassignment"][observed], oob_counts[observed] / requested[observed]
    )
    assert np.isnan(s.arrays["oob_coassignment"][~observed]).all()
    for k, count in enumerate(s.exact_recovery_counts):
        target = set(np.flatnonzero(result.point.labels == k))
        expected = sum(
            any(
                set(np.flatnonzero(np.array(r["labels"]) == c)) == target
                for c in range(r["selected_k"])
            )
            for r in s.replicates
        )
        assert count == expected


@pytest.mark.parametrize("all_failed", [False, True])
def test_failed_replicates_are_not_retried_or_removed(monkeypatch, all_failed):
    x, _ = simulate()
    original, calls = shape_gmm.fit_shape_pca, 0

    def fail(coordinates, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1 and (all_failed or calls == 2):
            raise ValueError("deliberate rank/convergence failure")
        return original(coordinates, **kwargs)

    monkeypatch.setattr(shape_gmm, "fit_shape_pca", fail)
    result = fit(x, settings(n_bootstrap=3))
    s = result.stability
    assert calls == 4 and s.requested == 3 and s.status == "unavailable"
    assert s.successful == (0 if all_failed else 2)
    assert np.isnan(s.arrays["coassignment"]).all()
    mask = ~np.eye(len(x), dtype=bool)
    np.testing.assert_allclose(
        (s.arrays["coassignment_upper"] - s.arrays["coassignment_lower"])[mask], len(s.failures) / 3
    )
    affected = s.arrays["oob_requested"] > s.arrays["oob_successful"]
    assert affected.any() and np.isnan(s.arrays["oob_coassignment"][affected]).all()


def test_dimension_independence_and_memory_guards():
    x, _ = simulate()
    with pytest.raises(ValueError, match="Repeated"):
        fit_shape_gmm(x, ["same"] * len(x), config=settings())
    with pytest.raises(ValueError, match="budget"):
        fit(x, settings(max_pairwise_bytes=1))
    with pytest.raises(ValueError, match="rank deficient"):
        fit(x, settings(n_pcs=9, min_cluster_size=10))
    with pytest.raises(ValueError, match="Too few"):
        fit(x[:2])
    with pytest.raises(ValueError, match="nonempty"):
        fit_shape_gmm(x, [" "] * len(x), config=settings())


def test_candidate_failure_aborts_the_declared_search():
    with pytest.raises(ValueError, match="Candidate K="):
        fit(simulate()[0], settings(max_iter=1, tol=1e-15))


def test_bootstrap_component_sizes_count_distinct_units():
    result = fit(simulate()[0])
    for record in result.stability.replicates:
        for c in record["candidates"]:
            if c["status"] == "eligible":
                assert min(c["distinct_unit_sizes"]) >= 5
                assert sum(c["hard_cluster_sizes"]) == 80
                assert sum(c["distinct_unit_sizes"]) == record["unique_training_units"]
