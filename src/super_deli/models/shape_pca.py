"""Centered covariance PCA in the orthogonal tangent plane at a fitted GPA consensus."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from super_deli.models.procrustes import align_preshapes, generalized_procrustes, preshapes
from super_deli.models.shape_config import ShapeConfig


@dataclass(frozen=True)
class ShapeProjection:
    aligned: np.ndarray
    centroids: np.ndarray
    centroid_sizes: np.ndarray
    rotations: np.ndarray
    tangent_coordinates: np.ndarray
    scores: np.ndarray
    angles_degrees: np.ndarray
    radial_shrinkage: np.ndarray
    reconstruction_error: np.ndarray


@dataclass(frozen=True)
class ShapePCAModel:
    config: ShapeConfig
    consensus: np.ndarray
    tangent_mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    total_variance: float
    numerical_rank: int
    shape_dimension: int


@dataclass(frozen=True)
class ShapePCAFit:
    model: ShapePCAModel
    projection: ShapeProjection
    history: tuple[dict, ...]


def _tangent(aligned: np.ndarray, consensus: np.ndarray, config: ShapeConfig):
    cosine = np.clip(np.einsum("nkd,kd->n", aligned, consensus), -1.0, 1.0)
    if np.any(cosine <= config.rank_tol):
        raise ValueError("Shape lies outside the open consensus hemisphere for tangent projection")
    tangent = aligned - cosine[:, None, None] * consensus
    angles = np.arccos(cosine)
    ratio = np.ones_like(angles)
    np.divide(np.sin(angles), angles, out=ratio, where=angles > 1e-12)
    return tangent, np.degrees(angles), 1 - ratio


def _projection(aligned, centers, sizes, rotations, model: ShapePCAModel) -> ShapeProjection:
    tangent, angles, shrinkage = _tangent(aligned, model.consensus, model.config)
    centered = tangent.reshape(len(tangent), -1) - model.tangent_mean
    scores = centered @ model.components.T
    error = np.linalg.norm(centered - scores @ model.components, axis=1)
    return ShapeProjection(
        aligned, centers, sizes, rotations, tangent, scores, angles, shrinkage, error
    )


def fit_shape_pca(coordinates, *, config: ShapeConfig | None = None) -> ShapePCAFit:
    config = config or ShapeConfig()
    if len(coordinates) < 2:
        raise ValueError("Shape PCA requires at least two retained observations")
    gpa = generalized_procrustes(coordinates, config=config)
    tangent, _, _ = _tangent(gpa.aligned, gpa.consensus, config)
    matrix = tangent.reshape(len(tangent), -1)
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    n, k, d = gpa.aligned.shape
    shape_dimension = k * d - d - 1 - d * (d - 1) // 2
    rank = min(
        int(np.sum(singular > config.rank_tol * max(float(singular[0]), 1.0))),
        n - 1,
        shape_dimension,
    )
    retained = min(config.n_components or rank, rank)
    components = components[:retained].copy()
    # Signs are reproducible when eigenvalues are distinct; tied PCA subspaces
    # have no unique axis basis and should be compared through their projection.
    if retained:
        pivots = np.argmax(np.abs(components), axis=1)
        components *= np.where(components[np.arange(retained), pivots] < 0, -1.0, 1.0)[:, None]
    total = float(np.sum(singular**2) / (n - 1)) if rank else 0.0
    variance = singular[:retained] ** 2 / (n - 1)
    ratios = variance / total if retained else np.empty(0)
    model = ShapePCAModel(
        config,
        gpa.consensus.copy(),
        mean,
        components,
        variance,
        ratios,
        total,
        rank,
        shape_dimension,
    )
    projection = _projection(gpa.aligned, gpa.centroids, gpa.centroid_sizes, gpa.rotations, model)
    return ShapePCAFit(model, projection, gpa.history)


def project_shape_pca(coordinates, *, model: ShapePCAModel) -> ShapeProjection:
    """Use a frozen reference/PC basis; do not update GPA, centering or loadings."""
    unit, centers, sizes = preshapes(coordinates, model.config)
    if unit.shape[1:] != model.consensus.shape:
        raise ValueError("Projection landmark count/dimensionality differs from the fitted model")
    aligned, rotations = align_preshapes(unit, model.consensus, model.config)
    return _projection(aligned, centers, sizes, rotations, model)
