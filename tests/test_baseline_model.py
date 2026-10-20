"""Bounded synthetic checks of behavior, not biological uncertainty calibration."""

# ruff: noqa: E402 -- optional dependencies must be checked before importing models.

from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from sklearn.metrics import adjusted_rand_score

from super_deli.models import pca_gmm
from super_deli.models.config import BaselineConfig
from super_deli.models.pca_gmm import (
    BaselineError,
    coassignment_counts,
    draw_block_indices,
    fit_pca_gmm,
)

pytestmark = pytest.mark.baseline


def simulate(kind="separated", sizes=(24, 24), seed=81):
    rng = np.random.default_rng(seed)
    n, loci = sum(sizes), 96
    labels = np.repeat([0, 1], sizes)
    if kind == "panmixia":
        p = np.broadcast_to(rng.uniform(0.15, 0.85, loci), (n, loci))
    elif kind == "cline":
        p = np.broadcast_to(np.linspace(0.1, 0.9, n)[:, None], (n, loci))
    else:
        difference = 0.38 if kind == "separated" else 0.06
        p = np.broadcast_to(
            np.where(labels[:, None] == 0, 0.5 - difference, 0.5 + difference), (n, loci)
        )
    return rng.binomial(2, p).astype(float), labels


def settings(**kwargs):
    return replace(
        BaselineConfig(
            2, "error", True, assume_unlinked=True, n_bootstrap=6, n_init=3, max_k=3, random_seed=71
        ),
        **kwargs,
    )


def fit(x, config=None, **kwargs):
    return fit_pca_gmm(x, [f"s{i}" for i in range(len(x))], config=config or settings(), **kwargs)


@pytest.mark.parametrize("sizes", [(24, 24), (12, 36)])
def test_separated_populations_and_unequal_sampling(sizes):
    x, truth = simulate(sizes=sizes)
    result = fit(x)
    assert result.point.selected_k == 2
    assert adjusted_rand_score(truth, result.point.labels) == 1
    support = result.stability.coassignment
    assert result.stability.status == "available"
    assert np.mean(support[np.equal.outer(truth, truth)]) > 0.7
    assert np.mean(support[~np.equal.outer(truth, truth)]) < 0.05
    partitions = [np.array(r["labels"]) for r in result.stability.replicates]
    np.testing.assert_allclose(support, coassignment_counts(partitions, len(x)) / 6)
    # Model selection can subdivide a generating population under imbalance.
    # Check the actual exact-set statistic instead of assuming perfect support.
    for k, count in enumerate(result.stability.exact_recovery_counts):
        target = set(np.flatnonzero(result.point.labels == k))
        expected = sum(
            any(set(np.flatnonzero(labels == c)) == target for c in set(labels))
            for labels in partitions
        )
        assert count == expected
    assert len(result.stability.replicates) == 6


def test_panmixia_keeps_one_cluster_candidate():
    result = fit(simulate("panmixia")[0], settings(n_bootstrap=2))
    assert result.point.selected_k == 1
    assert [c["k"] for c in result.point.candidates] == [1, 2, 3]
    assert np.all(result.point.membership == 1)


@pytest.mark.parametrize("kind", ["overlap", "cline"])
def test_overlap_and_continuous_structure_are_finite_without_truth_claims(kind):
    result = fit(simulate(kind)[0], settings(n_bootstrap=2))
    assert np.isfinite(result.point.membership).all()
    np.testing.assert_allclose(result.point.membership.sum(axis=1), 1)
    assert all(c.get("converged", False) for c in result.point.candidates)
    # The cline has no discrete species truth; a selected K is not a species count.
    assert 1 <= result.point.selected_k <= 3


def test_reproducibility_seed_and_exported_component_order():
    x, _ = simulate()
    before = x.copy()
    first, second = fit(x), fit(x)
    np.testing.assert_array_equal(x, before)
    np.testing.assert_array_equal(first.point.membership, second.point.membership)
    np.testing.assert_array_equal(first.stability.coassignment, second.stability.coassignment)
    assert first.stability.replicates == second.stability.replicates
    shorter = fit(x, settings(n_bootstrap=0))
    np.testing.assert_array_equal(first.point.membership, shorter.point.membership)
    np.testing.assert_allclose(
        first.point.membership,
        first.point.mixture.predict_proba(first.point.scores)[:, first.point.component_order],
    )
    assert shorter.stability.status == "not_computed"


def test_missingness_scaling_and_original_index_bookkeeping():
    x, _ = simulate()
    x[0, 0] = np.nan
    x[:, 1] = np.nan
    x[:, 2] = 1
    with pytest.raises(BaselineError, match="explicit mean"):
        fit(x, settings(n_bootstrap=0))
    for scale in ("center", "standard", "patterson"):
        result = fit(x, settings(n_bootstrap=0, missing_policy="mean", scaling=scale))
        pre = result.point.preprocessing
        assert pre.imputed_calls == 1
        assert pre.exclusions == (
            {"locus_index": 1, "reason": "all_missing"},
            {"locus_index": 2, "reason": "zero_dosage_variance"},
        )
        assert 1 not in pre.kept_indices and 2 not in pre.kept_indices
        assert np.isfinite(result.point.scores).all()
    x[0] = np.nan
    with pytest.raises(BaselineError, match="no observed"):
        fit(x, settings(missing_policy="mean"))


def test_linked_columns_resample_as_whole_unequal_blocks():
    groups = (np.array([0, 1, 2]), np.array([3]), np.array([4, 5]))
    draw = draw_block_indices(groups, np.random.default_rng(81))
    counts = np.bincount(draw, minlength=6)
    assert counts[0] == counts[1] == counts[2]
    assert counts[4] == counts[5]
    assert counts[0] + counts[3] + counts[4] == len(groups)
    x, _ = simulate()
    linked = np.repeat(x[:, :32], 3, axis=1)
    result = fit(
        linked,
        settings(assume_unlinked=False),
        block_ids=[str(i) for i in range(32) for _ in range(3)],
    )
    assert result.stability.block_count == 32
    assert result.stability.block_source == "provided"


@pytest.mark.parametrize(
    "blocks,unlinked",
    [
        (None, False),
        (["one"] * 96, False),
        (["b"] * 95, False),
        (["a", "b"] * 48, True),
        ([" "] * 96, False),
    ],
)
def test_invalid_or_absent_block_assumptions(blocks, unlinked):
    with pytest.raises(BaselineError):
        fit(simulate()[0], settings(assume_unlinked=unlinked), block_ids=blocks)


@pytest.mark.parametrize(
    "bad",
    [
        np.full((8, 4), 1.0),
        np.full((8, 4), np.nan),
        np.full((8, 4), 3.0),
        np.ones((2, 4)),
        np.full((8, 4), np.inf),
    ],
)
def test_uninformative_or_invalid_input(bad):
    with pytest.raises(BaselineError):
        fit(bad)


def test_repeat_specimens_memory_and_rank_guards():
    x, _ = simulate()
    with pytest.raises(BaselineError, match="Repeated"):
        fit_pca_gmm(x, ["repeat"] * len(x), config=settings())
    with pytest.raises(BaselineError, match="budget"):
        fit(x, settings(max_pairwise_bytes=20))
    with pytest.raises(BaselineError, match="rank deficient"):
        fit(np.repeat(x[:, :1], 10, axis=1))


def test_candidate_convergence_failure_does_not_silently_reduce_k_search():
    with pytest.raises(BaselineError, match="Candidate K="):
        fit(simulate("overlap")[0], settings(max_iter=1, tol=1e-15))


def test_small_components_and_infeasible_k_are_explicit():
    result = fit(simulate()[0], settings(n_bootstrap=0, min_cluster_size=24, max_k=3))
    assert result.point.candidates[-1]["status"] == "insufficient_samples"
    assert result.point.candidates[0]["status"] == "eligible"


def test_coassignment_is_invariant_to_label_permutation():
    a, b = np.array([0, 0, 1]), np.array([8, 8, 3])
    np.testing.assert_array_equal(coassignment_counts([a], 3), coassignment_counts([b], 3))


def test_failed_bootstrap_keeps_requested_denominator_and_withholds_point_estimate(monkeypatch):
    original = pca_gmm._fit_once
    calls = 0

    def fail_once(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise BaselineError("deliberate numerical failure")
        return original(*args)

    monkeypatch.setattr(pca_gmm, "_fit_once", fail_once)
    result = fit(simulate()[0], settings(n_bootstrap=3))
    s = result.stability
    assert s.status == "unavailable" and s.successful == 2 and s.requested == 3
    assert s.coassignment is None
    assert len(s.failures) == 1 and "fit_seed" in s.failures[0]
    mask = ~np.eye(48, dtype=bool)
    np.testing.assert_allclose((s.coassignment_upper - s.coassignment_lower)[mask], 1 / 3)
    assert sum(s.selected_k_counts.values()) == 2
    assert calls == 4  # one point fit, three attempts; no retry or exclusion.
