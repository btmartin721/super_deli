"""Inclusion-conditional coassignment with explicit noise and failed-draw bounds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SubsampleStability:
    summary: dict
    records: tuple[dict, ...]
    arrays: dict[str, np.ndarray]


def subsample_indices(n, count, fraction, seed):
    size = int(np.floor(n * fraction))
    if not 1 <= size < n:
        raise ValueError("Subsampling must include at least one and fewer than all units")
    return tuple(
        np.sort(np.random.default_rng(s).choice(n, size, replace=False))
        for s in np.random.SeedSequence(seed).spawn(count)
    )


def _ratio(a, b):
    return np.divide(a, b, out=np.full(a.shape, np.nan, dtype=float), where=b > 0)


def summarize_subsamples(n, records):
    """Failure bounds use attempted inclusion, never turn a failed draw into K=1."""
    included = np.zeros(n, dtype=np.int64)
    successful = included.copy()
    noise = included.copy()
    pair_included = np.zeros((n, n), dtype=np.int64)
    pair_successful = pair_included.copy()
    same = pair_included.copy()
    ks, noises = {}, {}
    for record in records:
        ids = np.asarray(record["training_indices"], dtype=int)
        if ids.ndim != 1 or len(set(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= n):
            raise ValueError("Subsample identities must be unique valid source indices")
        included[ids] += 1
        ix = np.ix_(ids, ids)
        pair_included[ix] += 1
        if record["status"] == "failed":
            continue
        if record["status"] != "complete":
            raise ValueError("Unknown subsample status")
        labels = np.asarray(record["labels"], dtype=int)
        if labels.shape != ids.shape or np.any(labels < -1):
            raise ValueError("Labels must match included units, with -1 for noise")
        successful[ids] += 1
        pair_successful[ix] += 1
        noise[ids] += labels == -1
        same[ix] += (labels[:, None] == labels[None, :]) & (labels[:, None] >= 0)
        k, z = len(set(labels) - {-1}), int(np.sum(labels < 0))
        ks[str(k)] = ks.get(str(k), 0) + 1
        noises[str(z)] = noises.get(str(z), 0) + 1
    if type(n) is not int or n < 1:
        raise ValueError("Positive source unit count required")
    failed_unit, failed_pair = included - successful, pair_included - pair_successful
    arrays = {
        "unit_included": included,
        "unit_successful": successful,
        "unit_noise": noise,
        "pair_included": pair_included,
        "pair_successful": pair_successful,
        "pair_coassigned": same,
        "noise_frequency": _ratio(noise, successful),
        "coassignment": _ratio(same, pair_successful),
        "noise_lower": _ratio(noise, included),
        "noise_upper": _ratio(noise + failed_unit, included),
        "coassignment_lower": _ratio(same, pair_included),
        "coassignment_upper": _ratio(same + failed_pair, pair_included),
    }
    for target, numerator, denominator in (
        ("noise_mc_se", noise, successful),
        ("coassignment_mc_se", same, pair_successful),
    ):
        probability = _ratio(numerator, denominator)
        arrays[target] = np.sqrt(_ratio(probability * (1 - probability), denominator))
    done = sum(r["status"] == "complete" for r in records)
    return SubsampleStability(
        {
            "schema_version": 1,
            "status": (
                "not_computed"
                if not records
                else "complete"
                if done == len(records)
                else "unavailable"
                if not done
                else "incomplete"
            ),
            "requested": len(records),
            "successful": done,
            "failed": len(records) - done,
            "k_frequencies": ks,
            "noise_count_frequencies": noises,
            "unit_coverage": float(np.mean(successful > 0)) if records else 0.0,
            "pair_coverage": float(np.mean(pair_successful[np.triu_indices(n, 1)] > 0))
            if records and n > 1
            else 0.0,
            "denominator": "successful unit/pair inclusions; arrays retain attempts",
            "failure_bounds": "all attempted inclusions; failed outcomes range from zero to one",
            "precision": "conditional Monte Carlo precision only; draws are repeated measurements",
            "interpretation": "positive-cluster coassignment; noise is never a shared cluster",
            "out_of_sample": "not available; no labels imputed for absent units",
        },
        tuple(records),
        arrays if records else {},
    )
