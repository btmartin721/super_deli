"""Independent moments, projection and whole-group refits for target-mean PCA."""

# ruff: noqa: E402
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from super_deli.benchmarking.grouped_study import simulate_grouped
from super_deli.io.config import read_json
from super_deli.models.grouped_config import GroupedCalibrationConfig
from super_deli.models.grouped_null import write_raw_views
from super_deli.models.integrative import IntegrationInput, fit_integrative, project_integrative
from super_deli.workflows.integrative import run_integrative

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.baseline


def prepared(level="locality", family="tied", draws=3):
    values, layout, _ = simulate_grouped("complementary", 64, 817, target_level=level)
    model = GroupedCalibrationConfig.from_dict(
        read_json(ROOT / "examples/shared-grouped-calibration/calibration.json")
    ).model
    model = replace(
        model,
        target_level=level,
        covariance_type=family,
        n_pcs=4,
        n_bootstrap=draws,
        views=(replace(model.views[0], pca_fit="target_means"), model.views[1]),
    )
    inputs = tuple(
        IntegrationInput(
            v["modality"],
            v["kind"],
            v["level"],
            values[v["kind"]],
            tuple(v["observation_ids"]),
            tuple(v["unit_ids"]),
            tuple(v["target_ids"]),
            tuple(v["feature_names"]),
            ("continuous",) * 4,
        )
        for v in layout["views"]
    )
    return inputs, model, layout


@pytest.mark.parametrize("level", ["population", "locality"])
@pytest.mark.parametrize("family", ["full", "tied"])
def test_equal_group_moments_projection_and_bootstrap_multiplicities(level, family):
    inputs, model, layout = prepared(level, family)
    before = inputs[0].values.copy()
    fit = fit_integrative(inputs, config=model)
    targets = np.repeat(np.arange(64), layout["counts"])
    means = np.stack([before[targets == i].mean(0) for i in range(64)])
    pre = fit.pipeline.views[0].model
    np.testing.assert_allclose(pre.means, means.mean(0), atol=1e-12)
    np.testing.assert_allclose(pre.scales, means.std(0), atol=1e-12)
    assert not np.allclose(pre.means, before.mean(0))
    np.testing.assert_allclose(
        project_integrative(fit.inputs, fit.unit_ids, fit.pipeline), fit.pipeline.scores, atol=1e-11
    )
    np.testing.assert_array_equal(before, inputs[0].values)
    assert fit.stability.successful == 3
    for row in fit.stability.replicates:
        indices = np.asarray(row["training_indices"])
        diagnostic = row["views"][0]
        assert diagnostic["pca_training_target_ids"] == [fit.unit_ids[i] for i in indices]
        assert diagnostic["pca_training_counts"] == [layout["counts"][i] for i in indices]
        np.testing.assert_allclose(
            diagnostic["pca_feature_means"], means[indices].mean(0), atol=1e-12
        )
        np.testing.assert_allclose(
            diagnostic["pca_feature_scales"], means[indices].std(0), atol=1e-12
        )
        counts = np.bincount(indices, minlength=64)
        np.testing.assert_array_equal(
            np.bincount(diagnostic["source_training_indices"], minlength=len(before)),
            counts[targets],
        )
    without = fit_integrative(inputs, config=replace(model, n_bootstrap=0))
    np.testing.assert_array_equal(without.pipeline.scores, fit.pipeline.scores)
    np.testing.assert_array_equal(without.point.labels, fit.point.labels)


def test_whole_member_replication_does_not_reweight_target_mean_pca():
    inputs, model, layout = prepared(draws=0)
    fit = fit_integrative(inputs, config=model)
    m = inputs[0]
    targets = np.repeat(np.arange(64), layout["counts"])
    repeats = np.where(targets % 3 == 0, 3, 1)
    indices = np.repeat(np.arange(len(targets)), repeats)
    replicated = replace(
        m,
        values=m.values[indices],
        unit_ids=tuple(f"new_specimen_{i}" for i in range(len(indices))),
        observation_ids=tuple(f"new_obs_{i}" for i in range(len(indices))),
        target_ids=tuple(m.target_ids[i] for i in indices),
    )
    other = fit_integrative((replicated, inputs[1]), config=model)
    np.testing.assert_allclose(
        fit.pipeline.views[0].model.means, other.pipeline.views[0].model.means, atol=1e-12
    )
    np.testing.assert_allclose(fit.pipeline.scores, other.pipeline.scores, atol=1e-10)


@pytest.mark.parametrize("invalid", ["missing", "counts"])
def test_incomplete_and_count_data_are_rejected(invalid):
    inputs, model, _ = prepared(draws=0)
    m = inputs[0]
    if invalid == "missing":
        values = m.values.copy()
        values[0, 0] = np.nan
        m = replace(m, values=values)
    else:
        m = replace(m, values=np.floor(np.abs(m.values)), feature_types=("count",) * 4)
    with pytest.raises(ValueError, match="complete continuous"):
        fit_integrative((m, inputs[1]), config=model)


def test_frozen_score_calibration_rejects_new_policy_before_null_draws(tmp_path):
    from super_deli.models.calibration_config import GaussianCalibrationConfig
    from super_deli.workflows.score_calibration import run_integrative_calibration

    inputs, model, layout = prepared(family="full", draws=0)
    dataset = write_raw_views({d.kind: d.values for d in inputs}, layout, tmp_path / "raw")
    run_integrative(
        dataset, model_config=model, base_dir=tmp_path / "raw", output_dir=tmp_path / "point"
    )
    calibration = GaussianCalibrationConfig.from_dict(
        read_json(ROOT / "examples/calibration/calibration.json")
    )
    with pytest.raises(ValueError, match="Target-mean PCA"):
        run_integrative_calibration(
            tmp_path / "point", config=calibration, output_dir=tmp_path / "calibration"
        )
    assert not (tmp_path / "calibration").exists()


def test_unseen_group_projection_uses_frozen_training_parameters():
    from super_deli.models.tabular_gmm import project_tabular_pca

    inputs, model, _ = prepared(draws=0)
    fit = fit_integrative(inputs, config=model)
    rng = np.random.default_rng(581)
    ids = tuple(f"new_group_{i}" for i in range(6))
    new = []
    blocks = []
    for j, (original, view) in enumerate(zip(inputs, fit.pipeline.views, strict=True)):
        counts = np.array([1, 2, 3, 4, 2, 1]) if j == 0 else np.ones(6, dtype=int)
        targets = np.repeat(np.arange(6), counts)
        values = rng.normal(size=(sum(counts), 4)) + 4
        new.append(
            replace(
                original,
                values=values,
                unit_ids=tuple(f"new_{j}_{i}" for i in range(sum(counts))) if j == 0 else ids,
                observation_ids=tuple(f"new_obs_{j}_{i}" for i in range(sum(counts))),
                target_ids=tuple(ids[i] for i in targets),
            )
        )
        means = np.stack([values[targets == i].mean(0) for i in range(6)])
        profiles = project_tabular_pca(means, model=view.model)
        blocks.append((profiles - view.mean) * np.sqrt(view.normalized_weight) / view.scale)
    expected = fit.pipeline.pca.transform(np.concatenate(blocks, axis=1))
    actual = project_integrative(tuple(new), ids, fit.pipeline)
    np.testing.assert_allclose(actual, expected, atol=1e-11)
