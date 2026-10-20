"""Complementary evidence, group means, paired resampling and numerical fusion checks."""

# ruff: noqa: E402

from collections import Counter
from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from sklearn.metrics import adjusted_rand_score

from super_deli.models import integrative
from super_deli.models.integrative import IntegrationInput, fit_integrative, project_integrative
from super_deli.models.integrative_config import IntegrativeConfig, ViewConfig
from super_deli.models.shape_config import ShapeConfig

pytestmark = pytest.mark.baseline


def simulate(all_views=False, grouped=False, null=False):
    rng = np.random.default_rng(71)
    n = 80
    labels = np.repeat(np.arange(4), 20)
    x, y = (labels // 2) * 8 - 4, (labels % 2) * 8 - 4
    values = {
        "morphological": rng.normal(0, 0.4, (n, 4)) + (0 if null else y[:, None]),
        "ecological": rng.normal(0, 0.4, (n, 3)) + (0 if null else x[:, None]),
    }
    if all_views:
        values["genomic"] = rng.binomial(
            2, np.broadcast_to(np.where(x[:, None] < 0, 0.12, 0.88), (n, 64))
        ).astype(float)
        base = np.array([[-1, -0.7], [0, -1], [1, -0.7], [1, 0.7], [0, 1], [-1, 0.7]])
        shape = np.broadcast_to(base, (n, 6, 2)).copy()
        shape[:, 4, 1] += np.where(y < 0, -0.16, 0.16) + rng.normal(0, 0.01, n)
        shape[:, 2:4, 0] += rng.normal(0, 0.015, (n, 1))
        shape += rng.normal(0, 0.001, shape.shape)
        values["morphometric"] = shape
    inputs, configs = [], []
    units = tuple(f"u{i}" for i in range(n))
    for kind, array in values.items():
        if grouped and kind != "ecological":
            array = np.repeat(array, 2, axis=0)
            targets = tuple(u for u in units for _ in range(2))
            source_ids = tuple(f"{kind}_{i}" for i in range(len(array)))
            level, aggregation = "specimen", "mean"
        else:
            source_ids, targets = units, units
            level, aggregation = ("population" if grouped else "specimen"), "none"
        options = (
            {"ploidy": 2}
            if kind == "genomic"
            else ({"shape": ShapeConfig()} if kind == "morphometric" else {})
        )
        configs.append(ViewConfig(kind, kind, aggregation=aggregation, **options))
        inputs.append(
            IntegrationInput(
                kind,
                kind,
                level,
                array,
                tuple(f"obs_{i}" for i in range(len(array))),
                source_ids,
                targets,
                tuple(f"f{i}" for i in range(array.shape[1]))
                if kind in {"morphological", "ecological"}
                else (),
                ("continuous",) * array.shape[1] if kind in {"morphological", "ecological"} else (),
            )
        )
    config = IntegrativeConfig(
        True,
        views=tuple(configs),
        target_level="population" if grouped else "specimen",
        max_k=4,
        n_init=3,
        n_bootstrap=3,
        random_seed=71,
    )
    return tuple(inputs), config, labels


@pytest.mark.parametrize("all_views,grouped", [(False, False), (True, False), (True, True)])
def test_complementary_structure_requires_integrated_evidence(all_views, grouped):
    inputs, config, truth = simulate(all_views, grouped)
    result = fit_integrative(inputs, config=config)
    assert result.point.selected_k == 4
    assert adjusted_rand_score(truth, result.point.labels) == 1
    assert result.stability.status == "available"
    same = np.equal.outer(truth, truth)
    assert result.stability.arrays["coassignment"][same].mean() > 0.9
    assert result.stability.arrays["coassignment"][~same].mean() < 0.1
    projected = project_integrative(result.inputs, result.unit_ids, result.pipeline)
    np.testing.assert_allclose(projected, result.pipeline.scores, atol=1e-10)
    np.testing.assert_allclose(
        result.point.membership,
        result.point.mixture.predict_proba(projected)[:, result.point.component_order],
        atol=1e-10,
    )


def test_normalization_and_joint_geometry_match_explicit_calculation():
    inputs, config, _ = simulate()
    config = replace(
        config, views=(replace(config.views[0], weight=3), config.views[1]), n_bootstrap=0
    )
    result = fit_integrative(inputs, config=config)
    blocks = []
    for view, weight in zip(result.pipeline.views, (0.75, 0.25), strict=True):
        assert view.normalized_weight == weight
        block = (
            (view.profiles - view.profiles.mean(axis=0))
            / np.sqrt(np.var(view.profiles, axis=0, ddof=1).sum())
            * np.sqrt(weight)
        )
        assert np.var(block, axis=0, ddof=1).sum() == pytest.approx(weight)
        blocks.append(block)
    np.testing.assert_allclose(result.pipeline.fused, np.concatenate(blocks, axis=1))
    assert result.stability.status == "not_computed"


def test_whole_view_scale_and_feature_duplication_do_not_dominate_fusion():
    inputs, config, _ = simulate()
    config = replace(
        config, n_bootstrap=0, views=(replace(config.views[0], scaling="center"), config.views[1])
    )
    first = fit_integrative(inputs, config=config)
    data = replace(
        inputs[0],
        values=np.tile(inputs[0].values * 1000, (1, 2)),
        feature_names=tuple(f"copy{i}" for i in range(8)),
        feature_types=("continuous",) * 8,
    )
    second = fit_integrative((data, inputs[1]), config=config)

    def distances(fused):
        return ((fused[:, None, :] - fused[None, :, :]) ** 2).sum(axis=2)

    np.testing.assert_allclose(
        distances(first.pipeline.fused), distances(second.pipeline.fused), atol=1e-10
    )
    assert adjusted_rand_score(first.point.labels, second.point.labels) == 1


def test_missing_views_fail_or_are_excluded_before_learning():
    inputs, config, _ = simulate()
    retained = replace(
        inputs[1],
        values=inputs[1].values[2:],
        observation_ids=inputs[1].observation_ids[2:],
        unit_ids=inputs[1].unit_ids[2:],
        target_ids=inputs[1].target_ids[2:],
    )
    inputs[0].values[:2] = 1e100
    with pytest.raises(ValueError, match="complete_cases"):
        fit_integrative((inputs[0], retained), config=config)
    result = fit_integrative(
        (inputs[0], retained), config=replace(config, missing_views="complete_cases", n_bootstrap=0)
    )
    assert len(result.unit_ids) == 78
    assert {e["unit_id"] for e in result.exclusions} == {"u0", "u1"}
    np.testing.assert_allclose(
        result.pipeline.views[0].model.means, inputs[0].values[2:].mean(axis=0)
    )


def test_paired_group_draws_keep_every_source_member_together():
    inputs, config, _ = simulate(True, True)
    result = fit_integrative(inputs, config=config)
    for record in result.stability.replicates:
        draw_counts = Counter(record["training_indices"])
        for data, view_record in zip(result.inputs, record["views"], strict=True):
            source_counts = np.bincount(
                view_record["source_training_indices"], minlength=len(data.values)
            )
            expected = [draw_counts[result.unit_ids.index(target)] for target in data.target_ids]
            np.testing.assert_array_equal(source_counts, expected)


def test_reproducibility_point_seed_and_input_safety():
    inputs, config, _ = simulate()
    saved = [d.values.copy() for d in inputs]
    first, second = (fit_integrative(inputs, config=config) for _ in range(2))
    assert first.stability.replicates == second.stability.replicates
    for data, before in zip(inputs, saved, strict=True):
        np.testing.assert_array_equal(data.values, before)
    short = fit_integrative(inputs, config=replace(config, n_bootstrap=0))
    np.testing.assert_array_equal(first.point.membership, short.point.membership)


def test_row_order_is_aligned_by_ids():
    inputs, config, truth = simulate()
    config = replace(config, n_bootstrap=0)
    order = np.random.default_rng(4).permutation(80)
    data = inputs[1]
    shuffled = replace(
        data,
        values=data.values[order],
        observation_ids=tuple(data.observation_ids[i] for i in order),
        unit_ids=tuple(data.unit_ids[i] for i in order),
        target_ids=tuple(data.target_ids[i] for i in order),
    )
    result = fit_integrative((inputs[0], shuffled), config=config)
    assert adjusted_rand_score(truth, result.point.labels) == 1


def test_single_group_remains_candidate():
    inputs, config, _ = simulate(null=True)
    result = fit_integrative(inputs, config=replace(config, n_bootstrap=0))
    assert result.point.selected_k == 1


def test_failure_bounds_keep_all_requested_draws(monkeypatch):
    inputs, config, _ = simulate()
    original, calls = integrative._fit_once, 0

    def fail_once(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("deliberate view refit failure")
        return original(*args)

    monkeypatch.setattr(integrative, "_fit_once", fail_once)
    result = fit_integrative(inputs, config=config)
    s = result.stability
    assert calls == 4 and s.successful == 2 and s.status == "unavailable"
    assert np.isnan(s.arrays["coassignment"]).all()
    mask = ~np.eye(80, dtype=bool)
    np.testing.assert_allclose(
        (s.arrays["coassignment_upper"] - s.arrays["coassignment_lower"])[mask], 1 / 3
    )


def test_cross_level_and_duplicate_identity_guards():
    inputs, config, _ = simulate()
    with pytest.raises(ValueError, match="upward"):
        fit_integrative((replace(inputs[0], level="locality"), inputs[1]), config=config)
    with pytest.raises(ValueError, match="Repeated"):
        fit_integrative((replace(inputs[0], unit_ids=("same",) * 80), inputs[1]), config=config)


def test_memory_rank_and_no_modality_drop_guards():
    inputs, config, _ = simulate()
    for name in ("max_working_bytes", "max_pairwise_bytes"):
        with pytest.raises(ValueError, match="exceeds"):
            fit_integrative(inputs, config=replace(config, **{name: 1}))
    constant = replace(inputs[0], values=np.ones_like(inputs[0].values))
    with pytest.raises(ValueError, match="Too few variable"):
        fit_integrative((constant, inputs[1]), config=config)
    with pytest.raises(ValueError, match="Candidate K="):
        fit_integrative(inputs, config=replace(config, max_iter=1, tol=1e-15))


def test_weight_normalization_is_stable_and_never_silently_discards_a_view():
    inputs, config, _ = simulate()
    huge = replace(
        config, n_bootstrap=0, views=tuple(replace(v, weight=1e308) for v in config.views)
    )
    result = fit_integrative(inputs, config=huge)
    assert all(v.normalized_weight == 0.5 for v in result.pipeline.views)
    tiny = replace(huge, views=(replace(huge.views[0], weight=1e-308), huge.views[1]))
    with pytest.raises(ValueError, match="float64 range"):
        fit_integrative(inputs, config=tiny)


def test_prepared_matrix_cannot_ignore_explicit_feature_selection():
    inputs, config, _ = simulate()
    config = replace(
        config, views=(replace(config.views[0], feature_names=("f0", "f1")), config.views[1])
    )
    with pytest.raises(ValueError, match="selected feature order"):
        fit_integrative(inputs, config=config)


def test_every_modality_pca_is_refitted_on_each_target_draw(monkeypatch):
    inputs, config, _ = simulate(True, True)
    calls = Counter()
    for name in ("fit_genomic_pca", "fit_tabular_pca", "fit_shape_pca"):
        original = getattr(integrative, name)

        def capture(*args, _function=original, _name=name, **kwargs):
            calls[_name] += 1
            return _function(*args, **kwargs)

        monkeypatch.setattr(integrative, name, capture)
    fit_integrative(inputs, config=config)
    assert calls == {"fit_genomic_pca": 4, "fit_shape_pca": 4, "fit_tabular_pca": 8}


def test_unequal_group_members_keep_explicit_source_observation_weights():
    inputs, config, _ = simulate(grouped=True)
    original = inputs[0]
    values = original.values[::2]
    counts = np.where(np.arange(80) % 3 == 0, 3, 1)
    targets = tuple(
        unit for unit, n in zip(original.target_ids[::2], counts, strict=True) for _ in range(n)
    )
    expanded = np.repeat(values, counts, axis=0)
    data = replace(
        original,
        values=expanded,
        target_ids=targets,
        observation_ids=tuple(f"obs{i}" for i in range(len(expanded))),
        unit_ids=tuple(f"specimen{i}" for i in range(len(expanded))),
    )
    result = fit_integrative((data, inputs[1]), config=replace(config, n_bootstrap=0))
    np.testing.assert_allclose(result.pipeline.views[0].model.means, expanded.mean(axis=0))


def test_oob_pair_denominators_are_recoverable_from_recorded_target_draws():
    inputs, config, _ = simulate()
    result = fit_integrative(inputs, config=config)
    requested = np.zeros((80, 80), dtype=int)
    same_counts = requested.copy()
    for record in result.stability.replicates:
        oob = np.zeros(80, dtype=bool)
        oob[record["oob_indices"]] = True
        pairs = np.logical_and.outer(oob, oob)
        labels = np.array(record["labels"])
        requested += pairs
        same_counts += pairs & np.equal.outer(labels, labels)
    np.testing.assert_array_equal(result.stability.arrays["oob_requested"], requested)
    np.testing.assert_array_equal(result.stability.arrays["oob_coassignment_counts"], same_counts)
