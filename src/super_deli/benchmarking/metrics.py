"""Truth-aware evaluation after fitting; independent datasets are the summary units."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from super_deli.io.config import read_json

METRIC_NAMES = (
    "ari",
    "k_correct",
    "absolute_k_error",
    "false_split",
    "pair_precision",
    "pair_recall",
    "pair_error",
    "retention_fraction",
    "bootstrap_success_fraction",
    "alternative_ari",
    "stable_cluster_exact_truth_fraction",
    "oob_pair_coverage",
    "oob_requested_pair_fraction",
    *(
        f"{scope}_{suffix}"
        for scope in ("coassignment", "oob_coassignment")
        for suffix in ("squared_error", "within_mean", "between_mean", "loss_lower", "loss_upper")
    ),
)


def _mean(values):
    return float(np.mean(values)) if len(values) else None


def _pair_scores(frequencies, truth, valid):
    p, y = frequencies[valid], truth[valid]
    return {
        "squared_error": _mean((p - y) ** 2),
        "within_mean": _mean(p[y]),
        "between_mean": _mean(p[~y]),
    }


def _loss_bounds(low, high, truth, valid):
    lower = np.where(truth, (1 - high) ** 2, low**2)
    upper = np.where(truth, (1 - low) ** 2, high**2)
    return _mean(lower[valid]), _mean(upper[valid])


def score_run(directory: Path, truth: dict) -> dict:
    """Align by IDs, exclude self-pairs, and retain unavailable stability as missing."""
    summary = read_json(directory / "summary.json")
    units = summary["unit_ids"]
    if len(set(units)) != len(units) or any(u not in truth for u in units):
        raise ValueError("Run identities must uniquely match generating truth")
    y = np.array([truth[u]["primary"] for u in units], dtype=int)
    with np.load(directory / "mixture.npz", allow_pickle=False) as arrays:
        labels = arrays["labels"]
    if labels.shape != y.shape:
        raise ValueError("Partition and target axes differ")
    n, selected_k = len(y), summary["selected_k"]
    pairs = np.triu(np.ones((n, n), dtype=bool), 1)
    same, predicted = np.equal.outer(y, y), np.equal.outer(labels, labels)
    tp = int(np.sum(pairs & same & predicted))
    fp = int(np.sum(pairs & ~same & predicted))
    fn = int(np.sum(pairs & same & ~predicted))
    k = len(set(y))
    metrics = {
        "ari": float(adjusted_rand_score(y, labels)),
        "k_correct": float(selected_k == k),
        "absolute_k_error": float(abs(selected_k - k)),
        "false_split": float(selected_k > 1) if k == 1 else None,
        "pair_precision": tp / (tp + fp) if tp + fp else None,
        "pair_recall": tp / (tp + fn) if tp + fn else None,
        "pair_error": (fp + fn) / int(pairs.sum()),
        "retention_fraction": n / len(truth),
        "bootstrap_success_fraction": summary["bootstrap"]["successful"]
        / summary["bootstrap"]["requested"],
    }
    alternative = [truth[u]["alternative"] for u in units]
    metrics["alternative_ari"] = (
        float(adjusted_rand_score(alternative, labels))
        if all(v is not None for v in alternative)
        else None
    )
    with np.load(directory / "stability.npz", allow_pickle=False) as arrays:
        frequency = arrays["coassignment"]
        oob = arrays["oob_coassignment"]
        for name, values in (("coassignment", frequency), ("oob_coassignment", oob)):
            valid = pairs & np.isfinite(values)
            for suffix, value in _pair_scores(values, same, valid).items():
                metrics[f"{name}_{suffix}"] = value
        requested = pairs & (arrays["oob_requested"] > 0)
        metrics["oob_pair_coverage"] = int(np.sum(pairs & np.isfinite(oob))) / int(pairs.sum())
        metrics["oob_requested_pair_fraction"] = int(requested.sum()) / int(pairs.sum())
        for prefix, mask in (("coassignment", pairs), ("oob_coassignment", requested)):
            low, high = _loss_bounds(
                arrays[f"{prefix}_lower"], arrays[f"{prefix}_upper"], same, mask
            )
            metrics[f"{prefix}_loss_lower"] = low
            metrics[f"{prefix}_loss_upper"] = high
        bins = []
        for name, values in (("all_pairs", frequency), ("oob_pairs", oob)):
            valid = pairs & np.isfinite(values)
            for j in range(10):
                mask = (
                    valid
                    & (values >= j / 10)
                    & ((values <= 1) if j == 9 else (values < (j + 1) / 10))
                )
                bins.append(
                    {
                        "scope": name,
                        "lower": j / 10,
                        "upper": (j + 1) / 10,
                        "pairs": int(mask.sum()),
                        "mean_frequency": _mean(values[mask]),
                        "true_same_fraction": _mean(same[mask]),
                    }
                )
    true_sets = {frozenset(np.flatnonzero(y == v)) for v in set(y)}
    clusters = []
    for c in read_json(directory / "clusters.json"):
        members = frozenset(i for i, u in enumerate(units) if u in set(c["unit_ids"]))
        clusters.append({**c, "matches_exact_truth_set": members in true_sets})
    stable = [c for c in clusters if c["frequency"] is not None and c["frequency"] >= 0.9]
    metrics["stable_cluster_exact_truth_fraction"] = _mean(
        [c["matches_exact_truth_set"] for c in stable]
    )
    generating_counts = Counter(int(v["primary"]) for v in truth.values())
    retained_counts = Counter(map(int, y))
    return {
        "metrics": metrics,
        "selected_k": selected_k,
        "truth_k_retained": k,
        "n_retained": n,
        "n_generating": len(truth),
        "generating_truth_counts": dict(sorted(generating_counts.items())),
        "retained_truth_counts": {k: retained_counts[k] for k in sorted(generating_counts)},
        "bootstrap": summary["bootstrap"],
        "point_seed": summary["point_seed"],
        "coassignment_bins": bins,
        "clusters": clusters,
        "stable_cluster_count": len(stable),
        "stable_wrong_cluster_count": sum(not c["matches_exact_truth_set"] for c in stable),
    }


def summarize_values(values, attempted):
    observed = [float(v) for v in values if v is not None]
    if any(not math.isfinite(v) for v in observed):
        raise ValueError("Summary metrics must be finite or explicitly missing")
    n = len(observed)
    return {
        "n_available": n,
        "n_missing": attempted - n,
        "mean": _mean(observed),
        "median": float(np.median(observed)) if n else None,
        "sd": float(np.std(observed, ddof=1)) if n > 1 else None,
        "mcse": float(np.std(observed, ddof=1) / np.sqrt(n)) if n > 1 else None,
        "min": float(min(observed)) if n else None,
        "max": float(max(observed)) if n else None,
    }


def summarize_runs(records, config):
    """Equal dataset weighting; pairs and inner bootstrap draws are not replicates."""
    groups, comparisons = [], []
    for scenario in config.scenarios:
        selected = [r for r in records if r["scenario"] == scenario.name]
        for arm in config.arms:
            runs = [r for r in selected if r["arm"] == arm.name]
            complete = [r for r in runs if r["status"] == "complete"]
            names = METRIC_NAMES
            groups.append(
                {
                    "scenario": scenario.name,
                    "arm": arm.name,
                    "attempted": len(runs),
                    "point_successful": len(complete),
                    "point_failed": len(runs) - len(complete),
                    "point_success_fraction": len(complete) / len(runs),
                    "point_success": summarize_values(
                        [float(r["status"] == "complete") for r in runs], len(runs)
                    ),
                    "max_rate_mcse_at_half": 0.5 / math.sqrt(len(runs)),
                    "selected_k_counts": dict(
                        Counter(str(r["evaluation"]["selected_k"]) for r in complete)
                    ),
                    "bootstrap_requested_including_failed_points": len(runs)
                    * config.model.n_bootstrap,
                    "bootstrap_successful": sum(
                        r["evaluation"]["bootstrap"]["successful"] for r in complete
                    ),
                    "metrics": {
                        k: summarize_values(
                            [r["evaluation"]["metrics"].get(k) for r in complete], len(runs)
                        )
                        for k in names
                    },
                }
            )
        reference = config.arms[0].name
        left = {r["repetition"]: r for r in selected if r["arm"] == reference}
        for arm in config.arms[1:]:
            right = {r["repetition"]: r for r in selected if r["arm"] == arm.name}
            names = METRIC_NAMES
            differences = {k: [] for k in names}
            for repetition in left:
                a, b = left[repetition], right[repetition]
                for k in names:
                    va = a.get("evaluation", {}).get("metrics", {}).get(k)
                    vb = b.get("evaluation", {}).get("metrics", {}).get(k)
                    differences[k].append(va - vb if va is not None and vb is not None else None)
            comparisons.append(
                {
                    "scenario": scenario.name,
                    "reference": reference,
                    "comparator": arm.name,
                    "direction": "reference minus comparator",
                    "attempted_pairs": len(left),
                    "metrics": {k: summarize_values(v, len(left)) for k, v in differences.items()},
                }
            )
    return {"groups": groups, "paired_comparisons": comparisons}
