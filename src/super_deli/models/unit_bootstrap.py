"""Whole-biological-unit pipeline bootstrap with label-invariant stability."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UnitStability:
    status: str
    requested: int
    successful: int
    failures: tuple[dict, ...]
    selected_k_counts: dict[int, int]
    exact_recovery_counts: tuple[int, ...]
    replicates: tuple[dict, ...]
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class BootstrapPrediction:
    labels: np.ndarray
    selected_k: int
    diagnostics: dict


def _bounds(counts, requested, successful):
    lower = np.full(counts.shape, np.nan)
    upper = lower.copy()
    np.divide(counts, requested, out=lower, where=requested > 0)
    np.divide(counts + requested - successful, requested, out=upper, where=requested > 0)
    frequency = np.where(requested == successful, lower, np.nan)
    return lower, upper, frequency


def bootstrap_units(n, point_labels, point_k, config, seeds, fit_predict):
    counts = np.zeros((n, n), dtype=np.uint64)
    oob_counts, oob_requested, oob_successful = (np.zeros_like(counts) for _ in range(3))
    recoveries = np.zeros(point_k, dtype=np.int64)
    groups = [frozenset(np.flatnonzero(point_labels == k)) for k in range(point_k)]
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
            prediction = fit_predict(indices, fit_seed)
            labels = np.asarray(prediction.labels)
            if (
                labels.shape != (n,)
                or labels.dtype.kind not in "iu"
                or type(prediction.selected_k) is not int
                or prediction.selected_k < 1
                or np.any(labels < 0)
                or np.any(labels >= prediction.selected_k)
            ):
                raise ValueError("Invalid bootstrap prediction partition")
        except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
            record.update(status="failed", error=str(exc))
            failures.append(record)
        else:
            same = np.equal.outer(labels, labels)
            counts += same
            oob_counts += same & oob_pairs
            oob_successful += oob_pairs
            observed = {
                frozenset(np.flatnonzero(labels == k)) for k in range(prediction.selected_k)
            }
            recoveries += [group in observed for group in groups]
            selected_k[prediction.selected_k] += 1
            record.update(prediction.diagnostics)
            record.update(
                status="complete", selected_k=prediction.selected_k, labels=labels.tolist()
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
    return UnitStability(
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
