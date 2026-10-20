"""Numeric PCA reference checks, known structure and full-pipeline unit bootstrap."""

# ruff: noqa: E402

from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from super_deli.core.data import Feature
from super_deli.core.tabular import TabularTable
from super_deli.models import tabular_gmm
from super_deli.models.tabular_cluster_config import TabularClusterConfig
from super_deli.models.tabular_gmm import fit_tabular_gmm, project_tabular_pca

pytestmark = pytest.mark.baseline


def table_from(x, *, kind="morphological", level="specimen", types=None):
    n, p = x.shape
    return TabularTable(
        "measurements",
        kind,
        level,
        tuple(Feature(f"v{i}", t) for i, t in enumerate(types or ["continuous"] * p)),
        tuple(f"obs{i}" for i in range(n)),
        tuple(f"unit{i}" for i in range(n)),
        tuple(tuple(None if np.isnan(v) else float(v) for v in row) for row in x),
    )


def simulate(kind="groups", seed=71):
    rng = np.random.default_rng(seed)
    truth = np.repeat([0, 1], 50)
    x = rng.normal(0, 1, (100, 4))
    if kind == "groups":
        x += (truth * 8 - 4)[:, None]
    elif kind == "cline":
        x += np.linspace(-6, 6, len(x))[:, None]
    return x, truth


def settings(**changes):
    return replace(
        TabularClusterConfig(True, n_bootstrap=5, n_init=3, max_k=3, random_seed=71), **changes
    )


@pytest.mark.parametrize(
    "kind,level",
    [
        ("morphological", "specimen"),
        ("ecological", "locality"),
        ("ecological", "population"),
    ],
)
def test_known_groups_and_level_preservation(kind, level):
    x, truth = simulate()
    result = fit_tabular_gmm(table_from(x, kind=kind, level=level), config=settings())
    assert result.point.selected_k == 2
    assert adjusted_rand_score(truth, result.point.labels) == 1
    assert result.point.mixture.covariance_type == "full"
    assert result.stability.status == "available"
    same = np.equal.outer(truth, truth)
    assert result.stability.arrays["coassignment"][same].mean() > 0.95
    assert result.stability.arrays["coassignment"][~same].mean() < 0.05


def test_single_group_and_continuous_cline_are_finite_without_species_claims():
    result = fit_tabular_gmm(table_from(simulate("null")[0]), config=settings(n_bootstrap=0))
    assert result.point.selected_k == 1
    assert result.point.candidates[0]["k"] == 1
    cline = fit_tabular_gmm(table_from(simulate("cline")[0]), config=settings(n_bootstrap=1))
    assert np.isfinite(cline.point.membership).all()
    assert 1 <= cline.point.selected_k <= 3


@pytest.mark.parametrize("scaling", ["standard", "center"])
@pytest.mark.parametrize("transform", ["identity", "log1p"])
def test_preprocessing_and_pca_match_independent_sklearn_reference(scaling, transform):
    x, _ = simulate()
    x[:, 3] = np.arange(len(x)) % 20
    data = table_from(x, types=("continuous", "continuous", "continuous", "count"))
    result = fit_tabular_gmm(
        data, config=settings(count_transform=transform, scaling=scaling, n_bootstrap=0)
    )
    transformed = x.copy()
    if transform == "log1p":
        transformed[:, 3] = np.log1p(transformed[:, 3])
    scaler = StandardScaler(with_std=scaling == "standard").fit(transformed)
    pca = PCA(n_components=2, svd_solver="full").fit(scaler.transform(transformed))
    np.testing.assert_allclose(result.model.means, scaler.mean_)
    np.testing.assert_allclose(
        result.model.scales, scaler.scale_ if scaler.scale_ is not None else np.ones(4)
    )
    np.testing.assert_allclose(result.model.pca.explained_variance_, pca.explained_variance_)
    np.testing.assert_allclose(project_tabular_pca(x, model=result.model), result.scores)
    np.testing.assert_allclose(
        result.point.membership,
        result.point.mixture.predict_proba(result.scores)[:, result.point.component_order],
    )


def test_missingness_exclusions_and_no_entirely_imputed_rows():
    x, _ = simulate()
    x[0, 0] = np.nan
    x = np.column_stack([x, np.ones(len(x)), np.full(len(x), np.nan)])
    with pytest.raises(ValueError, match="missing_policy"):
        fit_tabular_gmm(table_from(x), config=settings(n_bootstrap=0))
    result = fit_tabular_gmm(table_from(x), config=settings(missing_policy="mean", n_bootstrap=0))
    assert result.model.imputed_cells == 1
    assert result.model.exclusions == (
        {"feature": "v4", "reason": "zero_variance"},
        {"feature": "v5", "reason": "all_missing"},
    )
    observed_mean = np.nanmean(x[:, :4], axis=0)
    filled = np.where(np.isnan(x[:, :4]), observed_mean, x[:, :4])
    np.testing.assert_allclose(result.model.means, observed_mean)
    np.testing.assert_allclose(result.model.scales, filled.std(axis=0))
    x[0, :4] = np.nan
    with pytest.raises(ValueError, match="no observed"):
        fit_tabular_gmm(table_from(x), config=settings(missing_policy="mean"))


def test_bootstrap_refits_only_training_rows_and_projects_originals(monkeypatch):
    x, _ = simulate()
    x[0, 0] = np.nan
    original_fit, original_project = tabular_gmm._fit_once, tabular_gmm.project_tabular_pca
    draws, models, projected_models = [], [], []

    def capture_fit(values, *args):
        draws.append(values.copy())
        fitted = original_fit(values, *args)
        models.append(fitted.model)
        return fitted

    def capture_projection(values, *, model):
        np.testing.assert_array_equal(values, x)
        projected_models.append(model)
        return original_project(values, model=model)

    monkeypatch.setattr(tabular_gmm, "_fit_once", capture_fit)
    monkeypatch.setattr(tabular_gmm, "project_tabular_pca", capture_projection)
    result = fit_tabular_gmm(table_from(x), config=settings(missing_policy="mean"))
    assert len(draws) == 6 and len(projected_models) == 5
    for draw, model, record in zip(draws[1:], models[1:], result.stability.replicates, strict=True):
        np.testing.assert_array_equal(draw, x[record["training_indices"]])
        np.testing.assert_allclose(model.means, np.nanmean(draw, axis=0))
        assert model is not result.model
    assert all(a is b for a, b in zip(projected_models, models[1:], strict=True))


def test_seed_prefix_reproducibility_input_safety_and_units_invariance():
    x, _ = simulate()
    table = table_from(x)
    first, second = (fit_tabular_gmm(table, config=settings()) for _ in range(2))
    assert first.stability.replicates == second.stability.replicates
    np.testing.assert_array_equal(first.point.membership, second.point.membership)
    np.testing.assert_allclose(np.array(table.values), x)
    short = fit_tabular_gmm(table, config=settings(n_bootstrap=0))
    np.testing.assert_array_equal(short.point.membership, first.point.membership)
    moved = table_from(x * [1000, 0.1, 3, 42] + [6, 100, -100, 0])
    scaled = fit_tabular_gmm(moved, config=settings(n_bootstrap=0))
    np.testing.assert_allclose(scaled.point.membership, first.point.membership, atol=1e-7)


@pytest.mark.parametrize("all_failed", [False, True])
def test_failures_retain_denominators_and_oob_bounds(monkeypatch, all_failed):
    original, calls = tabular_gmm._fit_once, 0

    def fail(*args):
        nonlocal calls
        calls += 1
        if calls > 1 and (all_failed or calls == 2):
            raise ValueError("deliberate bootstrap failure")
        return original(*args)

    monkeypatch.setattr(tabular_gmm, "_fit_once", fail)
    result = fit_tabular_gmm(table_from(simulate()[0]), config=settings(n_bootstrap=3))
    s = result.stability
    assert calls == 4 and s.requested == 3 and s.status == "unavailable"
    assert s.successful == (0 if all_failed else 2)
    assert np.isnan(s.arrays["coassignment"]).all()
    mask = ~np.eye(100, dtype=bool)
    np.testing.assert_allclose(
        (s.arrays["coassignment_upper"] - s.arrays["coassignment_lower"])[mask], len(s.failures) / 3
    )
    observed = s.arrays["oob_requested"] > 0
    np.testing.assert_allclose(
        s.arrays["oob_coassignment_lower"][observed],
        s.arrays["oob_coassignment_counts"][observed] / s.arrays["oob_requested"][observed],
    )
    affected = s.arrays["oob_requested"] > s.arrays["oob_successful"]
    assert np.isnan(s.arrays["oob_coassignment"][affected]).all()


def test_covariate_and_categorical_features_require_explicit_selection():
    table = table_from(simulate()[0])
    features = (*table.features[:3], replace(table.features[3], dtype="categorical"))
    mixed = replace(table, features=features)
    with pytest.raises(ValueError, match="numeric feature_names"):
        fit_tabular_gmm(mixed, config=settings())
    selected = fit_tabular_gmm(mixed, config=settings(feature_names=("v2", "v0"), n_bootstrap=0))
    assert selected.model.feature_names == ("v2", "v0")
    covariate = replace(
        table, features=(*table.features[:3], replace(table.features[3], role="covariate"))
    )
    with pytest.raises(ValueError, match="Covariates"):
        fit_tabular_gmm(covariate, config=settings(feature_names=("v0", "v3")))
    with pytest.raises(ValueError, match="declared analytical"):
        fit_tabular_gmm(table, config=settings(feature_names=("unknown",)))


def test_repeated_units_memory_rank_and_numerical_guards():
    table = table_from(simulate()[0])
    with pytest.raises(ValueError, match="Repeated"):
        fit_tabular_gmm(replace(table, unit_ids=("same",) * 100), config=settings())
    for field in ("max_working_bytes", "max_pairwise_bytes"):
        with pytest.raises(ValueError, match="exceeds"):
            fit_tabular_gmm(table, config=settings(**{field: 1}))
    with pytest.raises(ValueError, match="rank deficient"):
        fit_tabular_gmm(
            table_from(np.repeat(np.arange(100)[:, None], 4, axis=1)), config=settings()
        )
    with pytest.raises(ValueError, match="Too few variable"):
        fit_tabular_gmm(table_from(np.ones((100, 4))), config=settings())
    huge = table_from(np.full((100, 4), 1e308))
    with pytest.raises(ValueError, match="Numerical"):
        fit_tabular_gmm(huge, config=settings())


@pytest.mark.parametrize("bad", [-1, 1.2, float("inf"), True, "3", 10**500, 2**53 + 1])
def test_invalid_counts(bad):
    table = table_from(simulate()[0])
    table = replace(
        table,
        features=(*table.features[:3], Feature("v3", "count")),
        values=tuple((*r[:3], bad) for r in table.values),
    )
    with pytest.raises(ValueError):
        fit_tabular_gmm(table, config=settings())


def test_candidate_convergence_failure_does_not_reduce_search():
    with pytest.raises(ValueError, match="Candidate K="):
        fit_tabular_gmm(table_from(simulate()[0]), config=settings(max_iter=1, tol=1e-15))


def test_projection_rejects_integer_precision_loss():
    x, _ = simulate()
    x[:, -1] = np.arange(len(x))
    table = table_from(x, types=("continuous", "continuous", "continuous", "count"))
    result = fit_tabular_gmm(table, config=settings(n_bootstrap=0))
    projected = np.ones((2, 4), dtype=np.uint64)
    projected[:, -1] = 2**53 + 1
    with pytest.raises(ValueError, match="exactly representable"):
        project_tabular_pca(projected, model=result.model)
