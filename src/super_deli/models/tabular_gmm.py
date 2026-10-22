"""Numeric morphology/ecology PCA/GMM with preprocessing refitted in each bootstrap."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from super_deli.core.tabular import TabularTable
from super_deli.models.mixture import FIT_LOCK, MixtureFit, select_gaussian_mixture
from super_deli.models.tabular_cluster_config import TabularClusterConfig
from super_deli.models.unit_bootstrap import BootstrapPrediction, UnitStability, bootstrap_units


@dataclass(frozen=True)
class TabularPCAModel:
    config: TabularClusterConfig
    feature_names: tuple[str, ...]
    feature_types: tuple[str, ...]
    kept_indices: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    exclusions: tuple[dict, ...]
    imputed_cells: int
    training_missing_fraction: np.ndarray
    pca: PCA


@dataclass(frozen=True)
class TabularPipelineFit:
    model: TabularPCAModel
    scores: np.ndarray
    mixture: MixtureFit


@dataclass(frozen=True)
class TabularMixtureFit:
    point: MixtureFit
    model: TabularPCAModel
    scores: np.ndarray
    stability: UnitStability
    unit_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    config: TabularClusterConfig
    point_seed: int


def _matrix(table, config):
    if table.kind not in {"morphological", "ecological"}:
        raise ValueError("Select a morphological or ecological table")
    n, p = len(table.values), len(table.features)
    if not n or any(len(row) != p for row in table.values):
        raise ValueError("Table rows must match the declared feature schema")
    for ids in (table.unit_ids, table.observation_ids):
        if len(ids) != n or any(not isinstance(u, str) or not u or u != u.strip() for u in ids):
            raise ValueError("One nonempty, unpadded unit and observation ID is required per row")
        if len(set(ids)) != len(ids):
            raise ValueError("Repeated units/observations must be aggregated before clustering")
    if table.level not in {"specimen", "population", "locality"}:
        raise ValueError("Invalid biological level")
    by_name = {f.name: i for i, f in enumerate(table.features)}
    if len(by_name) != p:
        raise ValueError("Duplicate feature names")
    names = config.feature_names or tuple(f.name for f in table.features if f.role == "feature")
    if not names or set(names) - by_name.keys():
        raise ValueError("Select at least one declared analytical feature")
    selected = tuple(table.features[by_name[name]] for name in names)
    if any(f.role != "feature" for f in selected):
        raise ValueError("Covariates cannot enter clustering; declare analytical roles explicitly")
    if any(f.dtype not in {"continuous", "count"} for f in selected):
        raise ValueError("PCA supports continuous/count features; select numeric feature_names")
    # Budget before creating the dense float matrix or fitting an SVD.
    q = min(n, len(names))
    estimate = 8 * (20 * n * len(names) + 6 * q * q + 6 * len(names))
    if estimate > config.max_working_bytes:
        raise ValueError("Estimated PCA workspace exceeds max_working_bytes")
    if config.n_bootstrap and 128 * n * n > config.max_pairwise_bytes:
        raise ValueError("Dense bootstrap stability exceeds max_pairwise_bytes budget")
    if n <= config.n_pcs or n < config.min_cluster_size:
        raise ValueError("Too few independent units for the requested PCA/mixture settings")
    values = [[row[by_name[name]] for name in names] for row in table.values]
    if any(v is not None and type(v) not in (int, float) for row in values for v in row):
        raise ValueError("Numeric features require finite real values or None missing cells")
    try:
        x = np.array([[np.nan if v is None else v for v in row] for row in values], dtype=float)
    except (OverflowError, ValueError) as exc:
        raise ValueError("Numeric values exceed finite float64 representation") from exc
    if any(
        v is not None and not np.isfinite(x[i, j])
        for i, row in enumerate(values)
        for j, v in enumerate(row)
    ):
        raise ValueError("Numeric features require finite values or None missing cells")
    for i, f in enumerate(selected):
        observed = x[~np.isnan(x[:, i]), i]
        if f.dtype == "count" and (np.any(observed < 0) or np.any(observed != np.floor(observed))):
            raise ValueError("Count features require nonnegative integer values")
        if f.dtype == "count" and any(
            row[i] is not None and row[i] != float(x[j, i]) for j, row in enumerate(values)
        ):
            raise ValueError("Count values must be exactly representable as float64")
    return x, tuple(names), tuple(f.dtype for f in selected)


def _transform_counts(x, types, config):
    result = np.array(x, dtype=float, copy=True)
    if config.count_transform == "log1p":
        indices = [i for i, dtype in enumerate(types) if dtype == "count"]
        result[:, indices] = np.log1p(result[:, indices])
    return result


def project_tabular_pca(values, *, model: TabularPCAModel) -> np.ndarray:
    """Project numeric columns in model.feature_names order using training parameters."""
    raw = np.asarray(values)
    if raw.dtype.kind not in "fiu":
        raise ValueError("Projection values must be real numeric data")
    x = np.asarray(raw, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(model.feature_names) or np.isinf(x).any():
        raise ValueError("Projection must match selected feature order with finite values or NaN")
    for i, dtype in enumerate(model.feature_types):
        if dtype == "count":
            observed = x[~np.isnan(x[:, i]), i]
            if np.any(observed < 0) or np.any(observed != np.floor(observed)):
                raise ValueError("Projection count features require nonnegative integers")
            if raw.dtype.kind in "iu" and any(
                int(raw[j, i]) != float(x[j, i]) for j in range(len(x))
            ):
                raise ValueError("Projection counts must be exactly representable as float64")
    x = _transform_counts(x, model.feature_types, model.config)[:, model.kept_indices]
    if np.any(np.isnan(x).all(axis=1)):
        raise ValueError("A projected unit has no observed values at retained features")
    if np.isnan(x).any() and model.config.missing_policy == "error":
        raise ValueError("Missing values require missing_policy='mean'")
    x = np.where(np.isnan(x), model.means, x)
    scores = model.pca.transform((x - model.means) / model.scales)
    if not np.isfinite(scores).all():
        raise ValueError("Projection produced nonfinite PCA scores")
    return scores


def fit_tabular_pca(x, names, types, config):
    """Fit preprocessing/PCA on a prepared analytical matrix, without a mixture."""
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        transformed = _transform_counts(x, types, config)
        observed = np.sum(~np.isnan(transformed), axis=0)
        means = np.divide(
            np.nansum(transformed, axis=0), observed, out=np.zeros(x.shape[1]), where=observed > 0
        )
        centered = np.where(np.isnan(transformed), 0, transformed - means)
        # Mean-filled values have zero centered residual. Their contribution
        # stays in the denominator used for standardization (ddof=0).
        variance = np.mean(centered**2, axis=0)
        keep = (observed > 0) & (variance > 0)
        kept = np.flatnonzero(keep)
        exclusions = tuple(
            {"feature": name, "reason": "all_missing" if observed[i] == 0 else "zero_variance"}
            for i, name in enumerate(names)
            if not keep[i]
        )
        if len(kept) < config.n_pcs:
            raise ValueError("Too few variable, observed features for the fixed PCA dimension")
        missing = np.isnan(transformed[:, kept])
        if missing.all(axis=1).any():
            raise ValueError("A unit has no observed values at retained features")
        if missing.any() and config.missing_policy == "error":
            raise ValueError("Missing values require missing_policy='mean'")
        scales = np.sqrt(variance[kept]) if config.scaling == "standard" else np.ones(len(kept))
        matrix = centered[:, kept] / scales
        pca = PCA(n_components=config.n_pcs, svd_solver="full", whiten=False)
        pca.fit(matrix)
        if pca.singular_values_[-1] <= config.rank_tol * max(pca.singular_values_[0], 1.0):
            raise ValueError("Requested fixed PCA dimension is rank deficient")
        scores = pca.transform(matrix)
    model = TabularPCAModel(
        config,
        names,
        types,
        kept,
        means[kept],
        scales,
        exclusions,
        int(missing.sum()),
        missing.mean(axis=1),
        pca,
    )
    return model, scores


def _fit_once(x, names, types, config, seed, indices=None):
    model, scores = fit_tabular_pca(x, names, types, config)
    mixture = select_gaussian_mixture(
        scores, config, seed, covariance_type="full", unit_indices=indices
    )
    return TabularPipelineFit(model, scores, mixture)


def fit_tabular_gmm(table: TabularTable, *, config: TabularClusterConfig) -> TabularMixtureFit:
    """Fit candidate numeric groups at the table's level; no cross-modality fusion."""
    config = TabularClusterConfig.from_dict(config.to_dict())
    x, names, types = _matrix(table, config)
    seeds = np.random.SeedSequence(config.random_seed).spawn(config.n_bootstrap + 1)
    point_seed = int(seeds[0].generate_state(1)[0])
    with FIT_LOCK, threadpool_limits(limits=config.n_threads):
        try:
            fitted = _fit_once(x, names, types, config, point_seed)
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"Numerical preprocessing/PCA failure: {exc}") from exc

        def fit_predict(indices, seed):
            refit = _fit_once(x[indices], names, types, config, seed, indices)
            scores = project_tabular_pca(x, model=refit.model)
            probabilities = refit.mixture.mixture.predict_proba(scores)
            if not np.isfinite(probabilities).all():
                raise ValueError("Nonfinite bootstrap membership probabilities")
            return BootstrapPrediction(
                probabilities.argmax(axis=1),
                refit.mixture.selected_k,
                {
                    "candidates": refit.mixture.candidates,
                    "retained_features": [names[i] for i in refit.model.kept_indices],
                    "excluded_features": refit.model.exclusions,
                    "imputed_cells": refit.model.imputed_cells,
                },
            )

        stability = (
            bootstrap_units(
                len(x),
                fitted.mixture.labels,
                fitted.mixture.selected_k,
                config,
                seeds[1:],
                fit_predict,
            )
            if config.n_bootstrap
            else UnitStability("not_computed", 0, 0, (), {}, (), (), {})
        )
    return TabularMixtureFit(
        fitted.mixture,
        fitted.model,
        fitted.scores,
        stability,
        tuple(table.unit_ids),
        tuple(table.observation_ids),
        config,
        point_seed,
    )
