"""Complete continuous tabular PCA/HDBSCAN with refitted subsampling stability."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import HDBSCAN
from threadpoolctl import threadpool_limits

from super_deli.models.hdbscan_config import HDBSCANConfig
from super_deli.models.mixture import FIT_LOCK
from super_deli.models.subsample_stability import (
    SubsampleStability,
    subsample_indices,
    summarize_subsamples,
)
from super_deli.models.tabular_gmm import TabularPCAModel, _matrix, fit_tabular_pca


@dataclass(frozen=True)
class DensityPoint:
    model: TabularPCAModel
    scores: np.ndarray
    labels: np.ndarray
    strength: np.ndarray

    @property
    def selected_k(self):
        return len(set(self.labels) - {-1})


@dataclass(frozen=True)
class DensityFit:
    point: DensityPoint
    stability: SubsampleStability
    unit_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    config: HDBSCANConfig


def prepare_density_table(table, config):
    """Reuse the existing table contract and reject unsupported analytical types."""
    x, names, types = _matrix(table, config.preprocessing_config())
    if set(types) != {"continuous"}:
        raise ValueError("HDBSCAN currently requires continuous analytical features")
    if not np.isfinite(x).all():
        raise ValueError("HDBSCAN currently requires complete finite values")
    # Includes a conservative quadratic distance workspace, even for tree paths.
    if 64 * len(x) ** 2 + 160 * x.size > config.max_working_bytes:
        raise ValueError("HDBSCAN distance/PCA workspace exceeds max_working_bytes")
    if config.n_subsamples and 112 * len(x) ** 2 > config.max_pairwise_bytes:
        raise ValueError("Subsampling support exceeds max_pairwise_bytes")
    if (
        config.n_subsamples
        and 96 * (x.size + len(x) * config.n_pcs) * (config.n_subsamples + 1)
        > config.max_working_bytes
    ):
        raise ValueError("Retained subsample fits exceed max_working_bytes")
    return x, names, types


def fit_density_once(x, names, types, config):
    """No prediction interface: labels and strengths belong only to training units."""
    model, scores = fit_tabular_pca(x, names, types, config.preprocessing_config())
    estimator = HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        metric="euclidean",
        alpha=config.alpha,
        cluster_selection_epsilon=float(config.cluster_selection_epsilon),
        cluster_selection_method=config.cluster_selection_method,
        allow_single_cluster=config.allow_single_cluster,
        n_jobs=1,
        copy=True,
    ).fit(scores)
    labels, strength = estimator.labels_, estimator.probabilities_
    if (
        np.any(labels < -1)
        or not np.isfinite(strength).all()
        or np.any((strength < 0) | (strength > 1))
        or np.any(strength[labels < 0] != 0)
    ):
        raise ValueError("Invalid density labels or membership strength")
    return DensityPoint(model, scores, labels.copy(), strength.copy())


def point_arrays(point):
    m = point.model
    return {
        "kept_indices": m.kept_indices,
        "means": m.means,
        "scales": m.scales,
        "pca_mean": m.pca.mean_,
        "components": m.pca.components_,
        "explained_variance": m.pca.explained_variance_,
        "explained_variance_ratio": m.pca.explained_variance_ratio_,
        "singular_values": m.pca.singular_values_,
        "scores": point.scores,
        "labels": point.labels,
        "strength": point.strength,
    }


def fit_tabular_hdbscan(table, *, config):
    config = HDBSCANConfig.from_dict(config.to_dict())
    x, names, types = prepare_density_table(table, config)
    arrays, records = {}, []
    with FIT_LOCK, threadpool_limits(limits=1):
        try:
            point = fit_density_once(x, names, types, config)
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"Numerical HDBSCAN/PCA failure: {exc}") from exc
        for i, indices in enumerate(
            subsample_indices(
                len(x), config.n_subsamples, config.subsample_fraction, config.random_seed
            )
            if config.n_subsamples
            else ()
        ):
            record = {
                "draw": i,
                "training_indices": indices.tolist(),
                "unit_ids": [table.unit_ids[j] for j in indices],
            }
            try:
                refit = fit_density_once(x[indices], names, types, config)
                arrays.update({f"draw_{i}_{key}": val for key, val in point_arrays(refit).items()})
                record.update(
                    status="complete",
                    labels=refit.labels.tolist(),
                    selected_k=refit.selected_k,
                    excluded_features=list(refit.model.exclusions),
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                record.update(status="failed", error=str(exc))
            records.append(record)
    stability = summarize_subsamples(len(x), records)
    stability.arrays.update(arrays)
    return DensityFit(point, stability, tuple(table.unit_ids), tuple(table.observation_ids), config)
