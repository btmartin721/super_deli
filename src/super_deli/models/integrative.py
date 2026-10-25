"""Weighted PC-score fusion with paired target-unit pipeline bootstraps."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from super_deli.models.integrative_config import IntegrativeConfig, ViewConfig
from super_deli.models.mixture import FIT_LOCK, MixtureFit, select_gaussian_mixture
from super_deli.models.pca_gmm import fit_genomic_pca, project_genomic_pca
from super_deli.models.shape_pca import fit_shape_pca, project_shape_pca
from super_deli.models.tabular_gmm import fit_tabular_pca, project_tabular_pca
from super_deli.models.unit_bootstrap import BootstrapPrediction, UnitStability, bootstrap_units


@dataclass(frozen=True)
class IntegrationInput:
    name: str
    kind: str
    level: str
    values: np.ndarray
    observation_ids: tuple[str, ...]
    unit_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    feature_names: tuple[str, ...] = ()
    feature_types: tuple[str, ...] = ()
    landmark_ids: tuple[str, ...] = ()
    original_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class ViewFit:
    config: ViewConfig
    model: object
    profiles: np.ndarray
    mean: np.ndarray
    scale: float
    normalized_weight: float
    training_indices: np.ndarray
    diagnostics: dict


@dataclass(frozen=True)
class IntegrationPoint:
    mixture: MixtureFit
    views: tuple[ViewFit, ...]
    fused: np.ndarray
    scores: np.ndarray
    pca: PCA


@dataclass(frozen=True)
class IntegrativeFit:
    point: MixtureFit
    pipeline: IntegrationPoint
    stability: UnitStability
    unit_ids: tuple[str, ...]
    exclusions: tuple[dict, ...]
    inputs: tuple[IntegrationInput, ...]
    config: IntegrativeConfig
    point_seed: int


def _prepare(inputs, config, *, fold_minimum=None):
    inputs = tuple(inputs)
    if len(inputs) != len(config.views):
        raise ValueError("One raw input is required for every selected view")
    sets = []
    for data, view in zip(inputs, config.views, strict=True):
        if (data.name, data.kind) != (view.modality, view.kind):
            raise ValueError("View identities/order differ from configuration")
        n = len(data.values)
        if not n or data.values.dtype.kind not in "fiu" or np.isinf(data.values).any():
            raise ValueError("Views require numeric arrays without infinities")
        for ids in (data.observation_ids, data.unit_ids, data.target_ids):
            if len(ids) != n or any(not isinstance(i, str) or not i or i != i.strip() for i in ids):
                raise ValueError("View rows require explicit unpadded source/target identities")
        if len(set(data.unit_ids)) != n or len(set(data.observation_ids)) != n:
            raise ValueError(
                "Repeated source units/observations require aggregation before ingestion"
            )
        if data.original_indices and (
            len(data.original_indices) != n
            or len(set(data.original_indices)) != n
            or any(type(i) is not int or i < 0 for i in data.original_indices)
        ):
            raise ValueError("Original source indices must uniquely cover retained rows")
        if data.level == config.target_level:
            if view.aggregation != "none" or data.target_ids != data.unit_ids:
                raise ValueError(
                    "Same-level integration requires exact unit IDs without aggregation"
                )
        elif not (
            data.level == "specimen"
            and config.target_level in {"population", "locality"}
            and view.aggregation == "mean"
        ):
            raise ValueError(
                "Cross-level integration requires explicit upward specimen mean aggregation"
            )
        if view.kind == "morphometric":
            if data.values.ndim != 3 or np.isnan(data.values).any():
                raise ValueError("Shape inputs require complete landmark configurations")
        elif data.values.ndim != 2:
            raise ValueError("Numeric views require row-by-feature matrices")
        if view.kind in {"morphological", "ecological"} and (
            len(data.feature_names) != data.values.shape[1]
            or len(set(data.feature_names)) != len(data.feature_names)
            or len(data.feature_types) != len(data.feature_names)
            or any(t not in {"continuous", "count"} for t in data.feature_types)
        ):
            raise ValueError("Tabular view arrays must match their numeric feature schema")
        if view.feature_names and data.feature_names != view.feature_names:
            raise ValueError("Prepared tabular arrays must respect the selected feature order")
        if view.kind in {"morphological", "ecological"}:
            for i, dtype in enumerate(data.feature_types):
                if dtype == "count":
                    observed = data.values[~np.isnan(data.values[:, i]), i]
                    if (
                        np.any(observed < 0)
                        or np.any(observed != np.floor(observed))
                        or any(
                            int(v) != float(v) for v in observed if data.values.dtype.kind in "iu"
                        )
                    ):
                        raise ValueError(
                            "Count view values require exactly representable nonnegative integers"
                        )
        if view.pca_fit == "target_means" and (
            data.level != "specimen"
            or any(t != "continuous" for t in data.feature_types)
            or not np.isfinite(data.values).all()
        ):
            raise ValueError("Target-mean PCA requires complete continuous specimen morphology")
        if (
            view.kind == "genomic"
            and not np.isin(data.values[~np.isnan(data.values)], [0, 1, 2]).all()
        ):
            raise ValueError("Genomic inputs require diploid 0/1/2 dosages or NaN")
        sets.append(set(data.target_ids))
    shared, union = set.intersection(*sets), set.union(*sets)
    exclusions = tuple(
        {
            "unit_id": unit,
            "missing_views": [
                v.modality for v, ids in zip(config.views, sets, strict=True) if unit not in ids
            ],
        }
        for unit in sorted(union - shared)
    )
    if exclusions and config.missing_views == "error":
        raise ValueError(
            "Entire missing views require missing_views='complete_cases'; no modality imputation"
        )
    units = tuple(dict.fromkeys(i for i in inputs[0].target_ids if i in shared))
    n = len(units)
    minimum = config.min_cluster_size
    if fold_minimum is not None:
        from super_deli.models.predictive_fold_size import PredictiveFoldMixture

        minimum = PredictiveFoldMixture.from_model(config, fold_minimum).min_cluster_size
    if n <= config.n_pcs or n < minimum:
        raise ValueError("Too few shared independent target units for integration")
    if config.n_bootstrap and 128 * n * n > config.max_pairwise_bytes:
        raise ValueError("Dense stability exceeds max_pairwise_bytes")
    # Remove nonshared units before fitting any numerical preprocessing.
    retained = []
    for data in inputs:
        rows = [i for i, target in enumerate(data.target_ids) if target in shared]
        source_indices = data.original_indices or tuple(range(len(data.values)))
        p = int(np.prod(data.values.shape[1:]))
        q = min(len(rows), p)
        if 8 * (20 * len(rows) * p + 6 * q * q + 6 * p) > config.max_working_bytes:
            raise ValueError("View PCA workspace exceeds max_working_bytes")
        retained.append(
            replace(
                data,
                values=np.array(data.values[rows], dtype=float, copy=True),
                observation_ids=tuple(data.observation_ids[i] for i in rows),
                unit_ids=tuple(data.unit_ids[i] for i in rows),
                target_ids=tuple(data.target_ids[i] for i in rows),
                original_indices=tuple(source_indices[i] for i in rows),
            )
        )
    return tuple(retained), units, exclusions


def _profiles(scores, targets, units):
    groups = {}
    for i, target in enumerate(targets):
        groups.setdefault(target, []).append(i)
    profiles = {unit: scores[groups[unit]].mean(axis=0) for unit in dict.fromkeys(units)}
    return np.stack([profiles[unit] for unit in units])


def _fit_view(data, view, units, draw, config, weight):
    # Every target draw contributes all its source observations, including
    # duplicated targets. Within-target observations are held fixed, not resampled.
    groups = {}
    for i, target in enumerate(data.target_ids):
        groups.setdefault(target, []).append(i)
    indices = np.concatenate([groups[units[i]] for i in draw])
    p = int(np.prod(data.values.shape[1:]))
    q = min(len(indices), p)
    if 8 * (20 * len(indices) * p + 6 * q * q + 6 * p) > config.max_working_bytes:
        raise ValueError("Bootstrap view workspace exceeds max_working_bytes")
    targets = tuple(data.target_ids[i] for i in indices)
    raw = data.values[indices]
    if view.kind == "genomic":
        scores, pre, pca = fit_genomic_pca(raw, view.genomic_config())
        model = (pre, pca)
        diagnostics = {"excluded_loci": pre.exclusions, "imputed_cells": pre.imputed_calls}
    elif view.kind == "morphometric":
        options = replace(
            view.shape,
            n_components=view.n_pcs,
            max_working_bytes=min(view.shape.max_working_bytes, config.max_working_bytes),
        )
        fitted = fit_shape_pca(raw, config=options)
        model, scores = fitted.model, fitted.projection.scores
        diagnostics = {
            "gpa_iterations": len(fitted.history),
            "numerical_rank": model.numerical_rank,
        }
    elif view.pca_fit == "target_means":
        drawn_units = [units[i] for i in draw]
        # One raw mean per target draw: keep multiplicities, never weight by member count.
        matrix = np.stack([data.values[groups[u]].mean(axis=0) for u in drawn_units])
        model, _ = fit_tabular_pca(
            matrix,
            data.feature_names,
            data.feature_types,
            view.tabular_config(config.max_working_bytes),
        )
        scores = project_tabular_pca(raw, model=model)
        diagnostics = {
            "excluded_features": model.exclusions,
            "imputed_cells": model.imputed_cells,
            "pca_fit": "target_means",
            "pca_training_target_ids": drawn_units,
            "pca_training_counts": [len(groups[u]) for u in drawn_units],
            "pca_feature_means": model.means.tolist(),
            "pca_feature_scales": model.scales.tolist(),
            "pca_kept_indices": model.kept_indices.tolist(),
            "pca_components": model.pca.components_.tolist(),
            "pca_center": model.pca.mean_.tolist(),
            "pca_explained_variance": model.pca.explained_variance_.tolist(),
        }
    else:
        model, scores = fit_tabular_pca(
            raw,
            data.feature_names,
            data.feature_types,
            view.tabular_config(config.max_working_bytes),
        )
        diagnostics = {"excluded_features": model.exclusions, "imputed_cells": model.imputed_cells}
    if scores.shape[1] != view.n_pcs:
        raise ValueError(f"{data.name}: fixed view PCA dimension is rank deficient")
    # Averaging duplicated source profiles is identical to their original
    # within-target mean. Duplicated target draws still enter joint PCA/GMM.
    profiles = _profiles(scores, targets, [units[i] for i in draw])
    mean = profiles.mean(axis=0)
    variance = np.sum((profiles - mean) ** 2) / (len(profiles) - 1)
    if not np.isfinite(variance) or variance <= config.rank_tol**2:
        raise ValueError(f"{data.name}: no resolved between-target score variance")
    return ViewFit(
        view, model, profiles, mean, float(np.sqrt(variance)), float(weight), indices, diagnostics
    )


def _project_view(data, fitted, units):
    if fitted.config.kind == "genomic":
        scores = project_genomic_pca(
            data.values, *fitted.model, missing_policy=fitted.config.missing_policy
        )
    elif fitted.config.kind == "morphometric":
        scores = project_shape_pca(data.values, model=fitted.model).scores
    else:
        scores = project_tabular_pca(data.values, model=fitted.model)
    profiles = _profiles(scores, data.target_ids, units)
    return (profiles - fitted.mean) * np.sqrt(fitted.normalized_weight) / fitted.scale


def _fit_once(inputs, units, config, seed, draw, *, fold_minimum=None):
    weights = np.array([v.weight for v in config.views], dtype=float)
    weights /= weights.max()
    weights /= weights.sum()
    if np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError("Relative weights exceed float64 range; no view is silently dropped")
    views = tuple(
        _fit_view(data, view, units, draw, config, w)
        for data, view, w in zip(inputs, config.views, weights, strict=True)
    )
    fused = np.concatenate(
        [(v.profiles - v.mean) * np.sqrt(v.normalized_weight) / v.scale for v in views], axis=1
    )
    n, p = fused.shape
    if 8 * (20 * n * p + 6 * min(n, p) ** 2 + 6 * p) > config.max_working_bytes:
        raise ValueError("Joint PCA workspace exceeds max_working_bytes")
    pca = PCA(n_components=config.n_pcs, svd_solver="full", whiten=False).fit(fused)
    if pca.singular_values_[-1] <= config.rank_tol * max(pca.singular_values_[0], 1.0):
        raise ValueError("Requested fixed joint PCA dimension is rank deficient")
    scores = pca.transform(fused)
    settings = config
    if fold_minimum is not None:
        from super_deli.models.predictive_fold_size import PredictiveFoldMixture

        settings = PredictiveFoldMixture.from_model(config, fold_minimum)
    mixture = select_gaussian_mixture(
        scores, settings, seed, covariance_type=config.covariance_type, unit_indices=draw
    )
    return IntegrationPoint(mixture, views, fused, scores, pca)


def project_integrative(inputs, units, pipeline):
    fused = np.concatenate(
        [
            _project_view(data, fitted, units)
            for data, fitted in zip(inputs, pipeline.views, strict=True)
        ],
        axis=1,
    )
    return pipeline.pca.transform(fused)


def fit_integrative(inputs, *, config: IntegrativeConfig):
    config = IntegrativeConfig.from_dict(config.to_dict())
    inputs, units, exclusions = _prepare(inputs, config)
    seeds = np.random.SeedSequence(config.random_seed).spawn(config.n_bootstrap + 1)
    point_seed = int(seeds[0].generate_state(1)[0])
    with FIT_LOCK, threadpool_limits(limits=config.n_threads):
        try:
            point = _fit_once(inputs, units, config, point_seed, np.arange(len(units)))
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"Numerical integration failure: {exc}") from exc

        def fit_predict(draw, seed):
            fitted = _fit_once(inputs, units, config, seed, draw)
            scores = project_integrative(inputs, units, fitted)
            probabilities = fitted.mixture.mixture.predict_proba(scores)
            if not np.isfinite(probabilities).all():
                raise ValueError("Nonfinite integrated bootstrap predictions")
            return BootstrapPrediction(
                probabilities.argmax(axis=1),
                fitted.mixture.selected_k,
                {
                    "candidates": fitted.mixture.candidates,
                    "views": [
                        {
                            "modality": v.config.modality,
                            "scale": v.scale,
                            "normalized_weight": v.normalized_weight,
                            "source_training_indices": v.training_indices.tolist(),
                            **v.diagnostics,
                        }
                        for v in fitted.views
                    ],
                },
            )

        stability = (
            bootstrap_units(
                len(units),
                point.mixture.labels,
                point.mixture.selected_k,
                config,
                seeds[1:],
                fit_predict,
            )
            if config.n_bootstrap
            else UnitStability("not_computed", 0, 0, (), {}, (), (), {})
        )
    return IntegrativeFit(
        point.mixture, point, stability, units, exclusions, inputs, config, point_seed
    )
