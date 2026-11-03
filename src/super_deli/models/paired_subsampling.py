"""Whole-ID subsampling shared by new paired four-coordinate workflows."""

import numpy as np

from super_deli.models import tabular_neural as native
from super_deli.models.matched_preprocessing import align_tables
from super_deli.models.subsample_stability import subsample_indices, summarize_subsamples


def fit_paired_subsamples(tables, *, config, point_function):
    x, alignment = align_tables(tables, config)
    if (
        192 * len(x) ** 2 > config.max_pairwise_bytes
        or 192 * len(x) ** 2 + 160 * x.size > config.max_working_bytes
        or 128 * (x.size + 4 * len(x) + 1000) * (config.n_subsamples + 1) > config.max_working_bytes
    ):
        raise ValueError("Retained separate-encoder support exceeds budget")
    point = point_function(x, config=config)
    records, saved = [], {}
    for i, ids in enumerate(
        subsample_indices(len(x), config.n_subsamples, 0.8, config.subsampling_seed)
        if config.n_subsamples
        else ()
    ):
        record = {
            "draw": i,
            "training_indices": ids.tolist(),
            "unit_ids": [alignment["unit_ids"][j] for j in ids],
            "seed": config.random_seed,
        }
        try:
            draw = point_function(x[ids], config=config)
            saved.update({f"draw_{i}_" + k: v for k, v in native.point_arrays(draw).items()})
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
                {f"draw_{i}_failed_" + k: v for k, v in getattr(exc, "arrays", {}).items()}
            )
        records.append(record)
    stable = summarize_subsamples(len(x), records)
    stable.arrays.update(saved)
    return point, stable, alignment
