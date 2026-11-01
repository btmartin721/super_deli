"""Denoising point and native subsampling fits with explicit failure evidence."""

from __future__ import annotations

import numpy as np
from threadpoolctl import threadpool_limits

from super_deli.models.denoising_config import DENOISING_TRAINING, DenoisingConfig
from super_deli.models.denoising_train import train_denoising
from super_deli.models.density_scores import fit_density_scores
from super_deli.models.mixture import FIT_LOCK
from super_deli.models.subsample_stability import subsample_indices, summarize_subsamples
from super_deli.models.tabular_hdbscan import prepare_density_table
from super_deli.models.tabular_neural import (
    NeuralFitError,
    NeuralModel,
    NeuralPoint,
    forward_numpy,
    latent_transform,
    point_arrays,
)


def fit_denoising_point(x, names, *, config):
    """Never accept truth labels, a seed search, or an oracle training target."""
    config = DenoisingConfig.from_dict(config.to_dict())
    x = np.asarray(x, dtype=float)
    if (
        x.ndim != 2
        or len(x) < 10
        or x.shape[1] != len(names)
        or tuple(names) != config.feature_names
        or not np.isfinite(x).all()
    ):
        raise ValueError("Denoising requires >=10 complete rows in declared feature order")
    if 64 * len(x) ** 2 + 160 * x.size > config.max_working_bytes:
        raise ValueError("Denoising distance/training workspace exceeds budget")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        means, scales = x.mean(0), x.std(0)
        kept = np.flatnonzero(scales > 0)
        if not 3 <= len(kept) <= 64:
            raise ValueError("Denoising requires 3–64 nonconstant features")
        scales[scales == 0] = 1
        standardized = (x[:, kept] - means[kept]) / scales[kept]
    with FIT_LOCK, threadpool_limits(limits=1):
        weights, losses, evidence = train_denoising(
            standardized, config.random_seed, config.corruption_seed
        )
        try:
            raw = forward_numpy(standardized, weights, "encoder")
            scores, lm, ls, singular = latent_transform(raw)
            labels, strength = fit_density_scores(scores, config.density_config())
        except Exception as exc:
            raise NeuralFitError(
                str(exc),
                diagnostics={
                    "completed_updates": 500,
                    "seed": config.random_seed,
                    "corruption_seed": config.corruption_seed,
                    "losses": losses.tolist(),
                },
                arrays={**weights, **evidence},
            ) from exc
    return NeuralPoint(
        NeuralModel(
            tuple(names),
            {
                **weights,
                **evidence,
                "means": means,
                "scales": scales,
                "kept_indices": kept,
                "latent_means": lm,
                "latent_scales": ls,
                "raw_embedding": raw,
            },
        ),
        scores,
        labels,
        strength,
        losses,
        {
            "training": DENOISING_TRAINING,
            "seed": config.random_seed,
            "corruption_seed": config.corruption_seed,
            "completed_updates": 500,
            "corruption_generator": "torch.Generator(cpu); float64 row-major randn each update",
            "latent_singular_values": singular.tolist(),
            "excluded_constant_features": [names[i] for i in range(len(names)) if i not in kept],
            "initial_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
            "species_significance": "not_computed",
        },
    )


def fit_tabular_denoising(table, *, config):
    config = DenoisingConfig.from_dict(config.to_dict())
    x, names, _ = prepare_density_table(table, config.density_config())
    widths = (x.shape[1], 16, 8, 2, 8, 16, x.shape[1])
    parameters = sum(a * b + b for a, b in zip(widths[:-1], widths[1:], strict=True))
    retained = (config.n_subsamples + 1) * (40 * parameters + 8 * 1001 + 256 * 500 + 16 * len(x))
    if retained + 64 * len(x) ** 2 + 160 * x.size > config.max_working_bytes:
        raise ValueError("Retained denoising evidence exceeds max_working_bytes")
    point = fit_denoising_point(x, names, config=config)
    records, arrays = [], {}
    for i, ids in enumerate(
        subsample_indices(len(x), config.n_subsamples, 0.8, config.subsampling_seed)
        if config.n_subsamples
        else ()
    ):
        record = {
            "draw": i,
            "training_indices": ids.tolist(),
            "unit_ids": [table.unit_ids[j] for j in ids],
            "seed": config.random_seed,
            "corruption_seed": config.corruption_seed,
        }
        try:
            draw = fit_denoising_point(x[ids], names, config=config)
            arrays.update({f"draw_{i}_{k}": v for k, v in point_arrays(draw).items()})
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
            arrays.update(
                {f"draw_{i}_failed_{k}": v for k, v in getattr(exc, "arrays", {}).items()}
            )
        records.append(record)
    stability = summarize_subsamples(len(x), records)
    stability.arrays.update(arrays)
    stability.summary["conditioning"] = (
        "fixed initialization/corruption seeds; no absent-unit predictions"
    )
    return point, stability
