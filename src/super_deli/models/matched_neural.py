"""View-balanced shared representation using the unchanged ordinary neural trainer."""

from dataclasses import dataclass

import numpy as np
from threadpoolctl import threadpool_limits

from super_deli.models.density_scores import fit_density_scores
from super_deli.models.matched_neural_config import MatchedNeuralConfig
from super_deli.models.matched_preprocessing import align_tables, prepare_views
from super_deli.models.mixture import FIT_LOCK
from super_deli.models.neural_config import TRAINING
from super_deli.models.subsample_stability import subsample_indices, summarize_subsamples
from super_deli.models.tabular_neural import (
    NeuralFitError,
    NeuralPoint,
    _train,
    forward_numpy,
    latent_transform,
    point_arrays,
)


@dataclass(frozen=True)
class MatchedNeuralModel:
    config: MatchedNeuralConfig
    arrays: dict

    def _weighted(self, tables):
        x, alignment = align_tables(tables, self.config, minimum=1)
        a = self.arrays
        kept = a["kept_indices"]
        return ((x[:, kept] - a["means"][kept]) / a["scales"][kept]) * a[
            "column_weights"
        ], alignment

    def encode(self, tables):
        weighted, alignment = self._weighted(tables)
        a = self.arrays
        z = (forward_numpy(weighted, a, "encoder") - a["latent_means"]) / a["latent_scales"]
        if not np.isfinite(z).all():
            raise ValueError("Nonfinite frozen paired encoding")
        return {"unit_ids": alignment["unit_ids"], "scores": z}

    def reconstruct(self, tables):
        weighted, alignment = self._weighted(tables)
        a = self.arrays
        decoded = (
            forward_numpy(forward_numpy(weighted, a, "encoder"), a, "decoder") / a["column_weights"]
        )
        result = np.broadcast_to(a["means"], (len(weighted), len(a["means"]))).copy()
        kept = a["kept_indices"]
        result[:, kept] = decoded * a["scales"][kept] + a["means"][kept]
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite frozen paired reconstruction")
        split = len(self.config.morphology_features)
        return {
            "unit_ids": alignment["unit_ids"],
            "views": {
                self.config.morphology: result[:, :split],
                self.config.ecology: result[:, split:],
            },
        }


def fit_matched_point(x, *, config):
    if (
        x.ndim != 2
        or x.shape[1] != len(config.morphology_features) + len(config.ecology_features)
        or len(x) < 10
    ):
        raise ValueError("Complete matched rows in declared joint feature order required")
    if 64 * len(x) ** 2 + 160 * x.size > config.max_working_bytes or not np.isfinite(x).all():
        raise ValueError("Invalid or excessive matched input workspace")
    weighted, preprocessing = prepare_views(x, len(config.morphology_features))
    with FIT_LOCK, threadpool_limits(limits=1):
        weights, losses = _train(weighted, config.random_seed)
        a = {**preprocessing, **weights}
        try:
            raw = forward_numpy(weighted, a, "encoder")
            scores, means, scales, singular = latent_transform(raw)
            a.update(latent_means=means, latent_scales=scales)
            labels, strength = fit_density_scores(scores, config.density_config())
        except Exception as exc:
            raise NeuralFitError(
                str(exc),
                arrays=a,
                diagnostics={
                    "seed": config.random_seed,
                    "completed_updates": 500,
                    "losses": losses.tolist(),
                },
            ) from exc
    residual = (forward_numpy(raw, a, "decoder") - weighted) / a["column_weights"]
    counts = a["retained_view_counts"]
    per_view = [
        float(np.mean(residual[:, : counts[0]] ** 2)),
        float(np.mean(residual[:, counts[0] :] ** 2)),
    ]
    diagnostics = {
        "training": TRAINING,
        "seed": config.random_seed,
        "completed_updates": 500,
        "latent_singular_values": singular.tolist(),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "per_view_standardized_mse": per_view,
        "view_weights": a["view_weights"].tolist(),
        "retained_view_counts": counts.tolist(),
        "species_significance": "not_computed",
    }
    return NeuralPoint(MatchedNeuralModel(config, a), scores, labels, strength, losses, diagnostics)


def fit_matched_neural(tables, *, config):
    config = MatchedNeuralConfig.from_dict(config.to_dict())
    x, alignment = align_tables(tables, config)
    # The native accumulator retains empty inclusion arrays even with zero
    # draws, so its quadratic workspace also needs a point-only guard.
    if (
        112 * len(x) ** 2 > config.max_pairwise_bytes
        or 112 * len(x) ** 2 + 160 * x.size > config.max_working_bytes
        or (
            config.n_subsamples
            and 96 * (x.size + 2 * len(x)) * (config.n_subsamples + 1) > config.max_working_bytes
        )
    ):
        raise ValueError("Retained matched subsampling support exceeds its declared budget")
    point = fit_matched_point(x, config=config)
    records, saved = [], {}
    for i, indices in enumerate(
        subsample_indices(len(x), config.n_subsamples, 0.8, config.subsampling_seed)
        if config.n_subsamples
        else ()
    ):
        record = {
            "draw": i,
            "training_indices": indices.tolist(),
            "unit_ids": [alignment["unit_ids"][j] for j in indices],
            "seed": config.random_seed,
        }
        try:
            draw = fit_matched_point(x[indices], config=config)
            saved.update({f"draw_{i}_{key}": value for key, value in point_arrays(draw).items()})
            record.update(
                status="complete",
                labels=draw.labels.tolist(),
                selected_k=draw.selected_k,
                diagnostics=draw.diagnostics,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            record.update(
                status="failed", error=str(exc), diagnostics=getattr(exc, "diagnostics", {})
            )
            saved.update(
                {
                    f"draw_{i}_failed_{key}": value
                    for key, value in getattr(exc, "arrays", {}).items()
                }
            )
        records.append(record)
    stability = summarize_subsamples(len(x), records)
    stability.arrays.update(saved)
    return point, stability, alignment
