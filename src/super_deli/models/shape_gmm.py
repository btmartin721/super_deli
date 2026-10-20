"""Full-covariance shape mixtures with whole-unit, full-pipeline bootstrap."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from threadpoolctl import threadpool_limits

from super_deli.models.mixture import FIT_LOCK, MixtureFit, select_gaussian_mixture
from super_deli.models.procrustes import coordinate_array
from super_deli.models.shape_cluster_config import ShapeClusterConfig
from super_deli.models.shape_pca import ShapePCAFit, fit_shape_pca, project_shape_pca


@dataclass(frozen=True)
class ShapeStability:
    status: str
    requested: int
    successful: int
    failures: tuple[dict, ...]
    selected_k_counts: dict[int, int]
    exact_recovery_counts: tuple[int, ...]
    replicates: tuple[dict, ...]
    arrays: dict[str, np.ndarray]


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


def _bounds(counts, requested, successful):
    lower = np.full(counts.shape, np.nan)
    upper = lower.copy()
    np.divide(counts, requested, out=lower, where=requested > 0)
    np.divide(counts + requested - successful, requested, out=upper, where=requested > 0)
    frequency = np.where(requested == successful, lower, np.nan)
    return lower, upper, frequency


def _bootstrap(x, point, config, seeds):
    n = len(x)
    counts = np.zeros((n, n), dtype=np.uint64)
    oob_counts, oob_requested, oob_successful = (np.zeros_like(counts) for _ in range(3))
    recoveries = np.zeros(point.selected_k, dtype=np.int64)
    groups = [frozenset(np.flatnonzero(point.labels == k)) for k in range(point.selected_k)]
    selected_k, records, failures = Counter(), [], []
    for i, child in enumerate(seeds):
        draw_seed, fit_seed = map(int, child.generate_state(2))
        indices = np.random.default_rng(draw_seed).integers(n, size=n)
        oob = np.bincount(indices, minlength=n) == 0
        oob_pairs = np.logical_and.outer(oob, oob)
        oob_requested += oob_pairs
        record = {
            "replicate": i,
            "draw_seed": draw_seed,
            "fit_seed": fit_seed,
            "training_indices": indices.tolist(),
            "unique_training_units": len(set(indices)),
            "oob_indices": np.flatnonzero(oob).tolist(),
        }
        try:
            # Raw landmark configurations, including bootstrap multiplicities,
            # refit the consensus, tangent plane, PCA, K search and mixture.
            shape = fit_shape_pca(x[indices], config=config.shape_config)
            mixture = _mixture(shape, config, fit_seed, indices)
            projection = project_shape_pca(x, model=shape.model)
            probabilities = mixture.mixture.predict_proba(projection.scores)
            if not np.isfinite(probabilities).all():
                raise ValueError("Nonfinite bootstrap prediction probabilities")
            labels = probabilities.argmax(axis=1)
        except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
            record.update(status="failed", error=str(exc))
            failures.append(record)
        else:
            same = np.equal.outer(labels, labels)
            counts += same
            oob_counts += same & oob_pairs
            oob_successful += oob_pairs
            observed = {frozenset(np.flatnonzero(labels == k)) for k in range(mixture.selected_k)}
            recoveries += [group in observed for group in groups]
            selected_k[mixture.selected_k] += 1
            record.update(
                status="complete",
                selected_k=mixture.selected_k,
                labels=labels.tolist(),
                candidates=mixture.candidates,
                gpa_iterations=len(shape.history),
                numerical_rank=shape.model.numerical_rank,
            )
        records.append(record)
    successful = config.n_bootstrap - len(failures)
    low, high, frequency = _bounds(counts, config.n_bootstrap, successful)
    # Self-coassignment is identically one, even when a fit fails.
    np.fill_diagonal(low, 1)
    np.fill_diagonal(high, 1)
    if not failures:
        np.fill_diagonal(frequency, 1)
    oob_low, oob_high, oob_frequency = _bounds(oob_counts, oob_requested, oob_successful)
    return ShapeStability(
        "unavailable" if failures else "available",
        config.n_bootstrap,
        successful,
        tuple(failures),
        dict(sorted(selected_k.items())),
        tuple(map(int, recoveries)),
        tuple(records),
        {
            "coassignment_counts": counts,
            "coassignment_lower": low,
            "coassignment_upper": high,
            "coassignment": frequency,
            "oob_coassignment_counts": oob_counts,
            "oob_requested": oob_requested,
            "oob_successful": oob_successful,
            "oob_coassignment_lower": oob_low,
            "oob_coassignment_upper": oob_high,
            "oob_coassignment": oob_frequency,
        },
    )


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
