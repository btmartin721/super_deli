"""Fixed whole-unit folds, Gaussian predictions and positive-gain screening."""

from __future__ import annotations

import math


def fold_assignments(unit_ids, config):
    import numpy as np

    units = tuple(unit_ids)
    if (
        len(units) < config.n_folds
        or len(set(units)) != len(units)
        or any(not isinstance(u, str) or not u or u != u.strip() for u in units)
    ):
        raise ValueError("Distinct unpadded target IDs must cover every fold")
    shuffled = np.random.default_rng(config.random_seed).permutation(len(units))
    rows = []
    for i, part in enumerate(np.array_split(shuffled, config.n_folds)):
        valid = sorted(map(int, part))
        train = sorted(set(range(len(units))) - set(valid))
        rows.append(
            {
                "fold": i,
                "training_indices": train,
                "validation_indices": valid,
                "training_unit_ids": [units[j] for j in train],
                "validation_unit_ids": [units[j] for j in valid],
            }
        )
    return rows


def gaussian_predictions(scores, arrays, k):
    """Log densities and component probabilities from already fitted parameters."""
    import numpy as np
    from scipy.special import logsumexp

    weights, means, covariances = (arrays[f"k{k}_{v}"] for v in ("weights", "means", "covariances"))
    if (
        scores.ndim != 2
        or not np.isfinite(scores).all()
        or np.any(weights <= 0)
        or not np.isclose(weights.sum(), 1)
        or len(weights) != k
    ):
        raise ValueError("Finite scores and valid fitted mixture parameters required")
    parts = []
    for w, mu, cov in zip(weights, means, covariances, strict=True):
        factor = np.linalg.cholesky(cov)
        z = np.linalg.solve(factor, (scores - mu).T).T
        parts.append(
            np.log(w)
            - 0.5
            * (
                scores.shape[1] * np.log(2 * np.pi)
                + 2 * np.log(np.diag(factor)).sum()
                + (z * z).sum(1)
            )
        )
    joint = np.column_stack(parts)
    log_density = logsumexp(joint, axis=1)
    probability = np.exp(joint - log_density[:, None])
    if not np.isfinite(log_density).all() or not np.isfinite(probability).all():
        raise ValueError("Nonfinite predictive density or membership")
    return log_density, probability


def screen_candidates(candidates, folds, *, n_units, n_folds):
    """No uncertainty claim: the fixed screen uses positive pooled validation gain."""
    eligible = [c for c in candidates if c["status"] == "eligible"]
    raw = min(eligible, key=lambda c: (c["aic"], c["k"]))["k"] if eligible else None
    bic = min(eligible, key=lambda c: (c["bic"], c["k"]))["k"] if eligible else None
    result = {
        "schema_version": 1,
        "policy": "positive_oof_gain_then_aic_v1",
        "ordinary_aic_k": raw,
        "descriptive_bic_k": bic,
        "screened_k": None,
        "status": "withheld",
        "candidates": [],
        "scope": "Predictive heuristic; no CV p-value, significance or species claim",
    }
    if not eligible or not any(c["k"] == 1 for c in eligible):
        return {**result, "error": "Unavailable full-data K=1 candidate"}
    if len(folds) != n_folds or sorted(f["fold"] for f in folds) != list(range(n_folds)):
        return {**result, "error": "Incomplete fold searches"}
    if any(f["status"] != "complete" for f in folds):
        return {**result, "error": "Numerical or pipeline failure in a retained fold"}
    indices = [j for f in folds for j in f["validation_indices"]]
    if sorted(indices) != list(range(n_units)):
        raise ValueError("Validation units must partition the input exactly once")
    for f in folds:
        if sorted(f["training_indices"] + f["validation_indices"]) != list(range(n_units)) or set(
            f["training_indices"]
        ) & set(f["validation_indices"]):
            raise ValueError("Training and validation partitions overlap or omit units")
        if not any(c["k"] == 1 and c["status"] == "eligible" for c in f["candidates"]):
            return {**result, "error": "Unavailable training-fold K=1 candidate"}
        for values in f["log_densities"].values():
            if len(values) != len(f["validation_indices"]) or not all(
                math.isfinite(x) for x in values
            ):
                return {**result, "error": "Unavailable finite validation densities"}
        if "1" not in f["log_densities"]:
            return {**result, "error": "Unavailable baseline validation density"}
    allowed = []
    for c in candidates:
        k = c["k"]
        states = [next((r["status"] for r in f["candidates"] if r["k"] == k), None) for f in folds]
        if any(s not in {"eligible", "small_component", "insufficient_samples"} for s in states):
            return {**result, "error": "Unresolved fold candidate"}
        gain = None
        if c["status"] == "eligible" and all(s == "eligible" for s in states):
            if any(str(k) not in f["log_densities"] for f in folds):
                return {**result, "error": "Missing eligible-candidate validation density"}
            gain = (
                math.fsum(
                    a - b
                    for f in folds
                    for a, b in zip(
                        f["log_densities"][str(k)], f["log_densities"]["1"], strict=True
                    )
                )
                / n_units
            )
        passes = k == 1 or (gain is not None and gain > 0)
        reason = (
            "baseline"
            if k == 1
            else "positive_gain"
            if passes
            else "ineligible_full_or_fold"
            if gain is None
            else "nonpositive_gain"
        )
        result["candidates"].append(
            {
                "k": k,
                "full_status": c["status"],
                "fold_statuses": states,
                "pooled_validation_gain": gain,
                "passes": passes,
                "reason": reason,
            }
        )
        if passes:
            allowed.append(c)
    result.update(status="complete", screened_k=min(allowed, key=lambda c: (c["aic"], c["k"]))["k"])
    return result
