"""Full-covariance shape mixtures with whole-unit, full-pipeline bootstrap."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from threadpoolctl import threadpool_limits

from super_deli.models.mixture import FIT_LOCK, MixtureFit, select_gaussian_mixture
from super_deli.models.procrustes import coordinate_array
from super_deli.models.shape_cluster_config import ShapeClusterConfig
from super_deli.models.shape_pca import ShapePCAFit, fit_shape_pca, project_shape_pca
from super_deli.models.unit_bootstrap import BootstrapPrediction, bootstrap_units
from super_deli.models.unit_bootstrap import UnitStability as ShapeStability


@dataclass(frozen=True)
class ShapeMixtureFit:
    shape: ShapePCAFit
    point: MixtureFit
    stability: ShapeStability
    unit_ids: tuple[str, ...]
    config: ShapeClusterConfig
    point_seed: int


def _validate(coordinates, unit_ids, config):
    config = ShapeClusterConfig.from_dict(config.to_dict())
    x = coordinate_array(coordinates, config.shape_config)
    ids = tuple(unit_ids)
    if len(ids) != len(x) or any(not isinstance(u, str) or not u or u != u.strip() for u in ids):
        raise ValueError("Supply one nonempty, unpadded biological unit ID per observation")
    if len(set(ids)) != len(ids):
        raise ValueError("Repeated biological units must be aggregated before shape clustering")
    if len(x) <= config.n_pcs or len(x) < config.min_cluster_size:
        raise ValueError("Too few independent units for the requested shape mixture")
    if config.n_bootstrap and 128 * len(x) ** 2 > config.max_pairwise_bytes:
        raise ValueError("Dense bootstrap stability exceeds max_pairwise_bytes budget")
    return x, ids


def _mixture(shape, config, seed, indices=None):
    if shape.projection.scores.shape[1] != config.n_pcs:
        raise ValueError("Requested fixed PCA dimension is rank deficient; no dimension reduction")
    return select_gaussian_mixture(
        shape.projection.scores, config, seed, covariance_type="full", unit_indices=indices
    )


def _bootstrap(x, point, config, seeds):
    def fit_predict(indices, fit_seed):
        shape = fit_shape_pca(x[indices], config=config.shape_config)
        mixture = _mixture(shape, config, fit_seed, indices)
        projection = project_shape_pca(x, model=shape.model)
        probabilities = mixture.mixture.predict_proba(projection.scores)
        if not np.isfinite(probabilities).all():
            raise ValueError("Nonfinite bootstrap prediction probabilities")
        return BootstrapPrediction(
            probabilities.argmax(axis=1),
            mixture.selected_k,
            {
                "candidates": mixture.candidates,
                "gpa_iterations": len(shape.history),
                "numerical_rank": shape.model.numerical_rank,
            },
        )

    return bootstrap_units(len(x), point.labels, point.selected_k, config, seeds, fit_predict)


def _cluster_fitted_shape(coordinates, unit_ids, *, config, shape):
    """Workflow bridge to reuse its already exported point GPA/PCA fit."""
    x, ids = _validate(coordinates, unit_ids, config)
    if shape.model.config != config.shape_config:
        raise ValueError("Point shape policies differ from the clustering configuration")
    seeds = np.random.SeedSequence(config.random_seed).spawn(config.n_bootstrap + 1)
    point_seed = int(seeds[0].generate_state(1)[0])
    with FIT_LOCK, threadpool_limits(limits=config.n_threads):
        point = _mixture(shape, config, point_seed)
        stability = (
            _bootstrap(x, point, config, seeds[1:])
            if config.n_bootstrap
            else ShapeStability("not_computed", 0, 0, (), {}, (), (), {})
        )
    return ShapeMixtureFit(shape, point, stability, ids, config, point_seed)


def fit_shape_gmm(coordinates, unit_ids, *, config: ShapeClusterConfig) -> ShapeMixtureFit:
    """Fit candidate shape groups; neither stability nor membership is species support."""
    x, ids = _validate(coordinates, unit_ids, config)
    with FIT_LOCK, threadpool_limits(limits=config.n_threads):
        shape = fit_shape_pca(x, config=config.shape_config)
        return _cluster_fitted_shape(x, ids, config=config, shape=shape)
