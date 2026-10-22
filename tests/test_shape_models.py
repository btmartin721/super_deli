"""Analytic geometry, similarity invariance, reflection and fixed-reference PCA."""

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
from super_deli.io.config import read_json  # noqa: E402
from super_deli.models.procrustes import GPAConvergenceError, generalized_procrustes  # noqa: E402
from super_deli.models.shape_config import ShapeConfig  # noqa: E402
from super_deli.models.shape_pca import fit_shape_pca, project_shape_pca  # noqa: E402

pytestmark = pytest.mark.shape
EXAMPLES = Path(__file__).resolve().parents[1] / "examples/shape"


def rectangles():
    return np.array(
        [[[-1, -h], [1, -h], [1, h], [-1, h]] for h in (0.35, 0.45, 0.55, 0.65, 0.75, 0.85)]
    )


def rotation2(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


@pytest.mark.parametrize("dimensions", [2, 3])
def test_translation_size_rotation_removal_and_reconstruction(dimensions):
    rng = np.random.default_rng(47)
    base = rng.normal(size=(6, dimensions))
    raw = []
    for i in range(5):
        r, _ = np.linalg.qr(rng.normal(size=(dimensions, dimensions)))
        r[:, -1] *= np.linalg.det(r)
        raw.append((base @ r) * (i + 1) + rng.normal(size=dimensions) * 20)
    raw = np.array(raw)
    before = raw.copy()
    result = generalized_procrustes(raw)
    np.testing.assert_allclose(raw, before)
    np.testing.assert_allclose(result.aligned.mean(axis=1), 0, atol=1e-13)
    np.testing.assert_allclose(np.linalg.norm(result.aligned, axis=(1, 2)), 1, atol=1e-13)
    np.testing.assert_allclose(
        result.aligned, np.broadcast_to(result.consensus, result.aligned.shape), atol=1e-12
    )
    restored = (result.aligned @ result.rotations.transpose(0, 2, 1)) * result.centroid_sizes[
        :, None, None
    ] + result.centroids[:, None, :]
    np.testing.assert_allclose(restored, raw, atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(result.rotations), 1, atol=1e-12)
    fit = fit_shape_pca(raw)
    assert fit.model.numerical_rank == 0 and fit.projection.scores.shape == (5, 0)


def test_analytic_rectangle_pca_and_tangent_geometry():
    fit = fit_shape_pca(rectangles())
    truth = read_json(EXAMPLES / "expected.json")
    assert fit.model.numerical_rank == truth["pca_rank"] == 1
    np.testing.assert_allclose(fit.model.consensus, truth["consensus"], atol=1e-12)
    np.testing.assert_allclose(
        fit.model.explained_variance, [truth["explained_variance"]], atol=1e-12
    )
    scores = fit.projection.scores[:, 0]
    expected = np.array(truth["scores_up_to_sign"])
    np.testing.assert_allclose(np.outer(scores, scores), np.outer(expected, expected), atol=1e-12)
    tangent = fit.projection.tangent_coordinates
    np.testing.assert_allclose(np.einsum("nkd,kd->n", tangent, fit.model.consensus), 0, atol=1e-13)
    np.testing.assert_allclose(scores.var(ddof=1), fit.model.explained_variance[0])
    np.testing.assert_allclose(fit.model.explained_variance_ratio, [1])
    assert np.max(fit.projection.reconstruction_error) < 1e-12
    objective = [h["sum_squared_residuals"] for h in fit.history]
    assert np.all(np.diff(objective) <= 1e-12)


def test_pairwise_closed_form_reference():
    # Two aligned unit configurations have tangent-PC variance 1 - cos(theta).
    scipy = pytest.importorskip("scipy.linalg")
    raw = rectangles()[[0, -1]]
    unit = raw / np.linalg.norm(raw, axis=(1, 2))[:, None, None]
    rotation, _ = scipy.orthogonal_procrustes(unit[1], unit[0])
    cosine = np.sum(unit[0] * (unit[1] @ rotation))
    fit = fit_shape_pca(raw)
    np.testing.assert_allclose(fit.model.explained_variance, [1 - cosine], atol=1e-12)


def test_frozen_projection_does_not_refit_and_is_batch_independent():
    fit = fit_shape_pca(rectangles())
    original = (
        fit.model.consensus.copy(),
        fit.model.components.copy(),
        fit.model.tangent_mean.copy(),
    )
    holdout = rectangles()[[1, 4]] * 9 + [100, -50]
    batch = project_shape_pca(holdout, model=fit.model)
    for i in range(2):
        single = project_shape_pca(holdout[i : i + 1], model=fit.model)
        np.testing.assert_allclose(single.scores[0], batch.scores[i], atol=1e-12)
    for before, after in zip(
        original, (fit.model.consensus, fit.model.components, fit.model.tangent_mean), strict=True
    ):
        np.testing.assert_array_equal(before, after)
    train = project_shape_pca(rectangles(), model=fit.model)
    np.testing.assert_allclose(train.scores, fit.projection.scores, atol=1e-12)


def test_similarity_and_observation_permutation_invariance_of_pca_kernel():
    raw = rectangles()
    altered = np.array(
        [x @ rotation2(0.6 * i) * (i + 1) + [i * 20, -i * 7] for i, x in enumerate(raw)]
    )
    fit = fit_shape_pca(raw)
    other = fit_shape_pca(altered)
    np.testing.assert_allclose(
        fit.projection.scores @ fit.projection.scores.T,
        other.projection.scores @ other.projection.scores.T,
        atol=1e-12,
    )
    order = [5, 2, 0, 4, 1, 3]
    shuffled = fit_shape_pca(altered[order])
    np.testing.assert_allclose(
        shuffled.model.explained_variance, fit.model.explained_variance, atol=1e-12
    )


def test_explicit_reflection_policy_preserves_or_removes_chirality():
    x = np.array([[0, 0], [3, 0], [2, 1], [0, 2]], dtype=float)
    pair = np.array([x, x * [-1, 1]])
    proper = fit_shape_pca(pair)
    reflected = fit_shape_pca(pair, config=ShapeConfig(reflection="allow"))
    assert proper.model.numerical_rank > 0 and reflected.model.numerical_rank == 0
    assert np.all(np.linalg.det(proper.projection.rotations) > 0)
    assert np.any(np.linalg.det(reflected.projection.rotations) < 0)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 4, 2)),
        np.ones((2, 4, 2)),
        np.full((2, 4, 2), np.nan),
        np.full((2, 4, 2), np.inf),
        np.ones((2, 2, 2)),
        np.ones((2, 4, 4)),
        np.ones((2, 4, 2), dtype=bool),
        np.ones((2, 4, 2), dtype=complex),
        np.array([[["1", "2"]] * 4] * 2),
    ],
)
def test_invalid_geometry_is_not_silently_coerced(bad):
    with pytest.raises(ValueError):
        fit_shape_pca(bad)


def test_3d_planar_is_supported_but_collinear_rotation_is_rejected():
    planar = np.pad(rectangles(), ((0, 0), (0, 0), (0, 1)))
    assert fit_shape_pca(planar).model.numerical_rank == 1
    line = np.array([[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]] * 2, dtype=float)
    with pytest.raises(ValueError, match="rank"):
        fit_shape_pca(line)


@pytest.mark.parametrize("scale", [1e-250, 1e250])
def test_extreme_finite_uniform_scaling_is_stable(scale):
    fit = fit_shape_pca(rectangles() * scale)
    np.testing.assert_allclose(
        fit.model.explained_variance,
        fit_shape_pca(rectangles()).model.explained_variance,
        atol=1e-12,
    )


def test_convergence_resource_limits_and_rank_clipping():
    with pytest.raises(GPAConvergenceError) as error:
        fit_shape_pca(rectangles(), config=ShapeConfig(max_iter=1))
    assert len(error.value.history) == 1
    with pytest.raises(ValueError, match="max_working_bytes"):
        fit_shape_pca(rectangles(), config=ShapeConfig(max_working_bytes=10))
    fit = fit_shape_pca(rectangles(), config=ShapeConfig(n_components=999))
    assert fit.model.components.shape == (1, 8)
    with pytest.raises(ValueError, match="at least two"):
        fit_shape_pca(rectangles()[:1])


def test_nonunique_proper_rotation_is_explicit():
    x = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    with pytest.raises(ValueError, match="nonunique"):
        fit_shape_pca(np.array([x, x * [-1, 1]]))


def test_outside_hemisphere_projection_is_rejected():
    fit = fit_shape_pca(rectangles())
    # This centered configuration is orthogonal in landmark space to both
    # coordinate columns of the fitted consensus, so its tangent chart is invalid.
    x = np.array([[[1, 0], [-1, 0], [1, 0], [-1, 0]]], dtype=float)
    with pytest.raises(ValueError):
        project_shape_pca(x, model=fit.model)
