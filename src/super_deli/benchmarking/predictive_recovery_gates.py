"""Diagnostic gate removals only; no fitting, p-values or production selector."""

import math


def decompose_gates(candidates, folds, n_units):
    result = {
        "role": "Diagnostic counterfactuals only; never production selection",
        "status": "withheld",
        "ordinary_k": None,
        "eligibility_only_k": None,
        "gain_only_k": None,
        "actual_k": None,
        "candidates": [],
    }
    full = [c for c in candidates if c["status"] == "eligible"]
    if (
        not any(c["k"] == 1 for c in full)
        or not folds
        or any(f["status"] != "complete" for f in folds)
    ):
        return {**result, "error": "Unavailable full baseline or incomplete folds"}
    if sorted(i for f in folds for i in f["validation_indices"]) != list(range(n_units)):
        raise ValueError("Validation folds do not partition the target units")
    for f in folds:
        if not any(c["k"] == 1 and c["status"] == "eligible" for c in f["candidates"]):
            return {**result, "error": "Unavailable fold baseline"}
    allowed = {k: [] for k in ("ordinary", "eligibility_only", "gain_only", "actual")}
    for c in candidates:
        k = c["k"]
        states = [next((v["status"] for v in f["candidates"] if v["k"] == k), None) for f in folds]
        if any(s not in {"eligible", "small_component", "insufficient_samples"} for s in states):
            return {**result, "error": "Unresolved numerical candidate"}
        gains = []
        fold_gains = []
        for f in folds:
            a, b = f["log_densities"].get(str(k)), f["log_densities"].get("1")
            if a is None or b is None:
                if states[folds.index(f)] != "insufficient_samples":
                    return {**result, "error": "Missing fitted validation density"}
                continue
            if (
                len(a) != len(f["validation_indices"])
                or len(b) != len(a)
                or not all(math.isfinite(x) for x in a + b)
            ):
                return {**result, "error": "Nonfinite or incomplete validation density"}
            delta = [x - y for x, y in zip(a, b, strict=True)]
            gains.extend(delta)
            fold_gains.append(math.fsum(delta) / len(delta))
        gain = math.fsum(gains) / n_units if len(gains) == n_units else None
        eligible = c["status"] == "eligible"
        every = all(s == "eligible" for s in states)
        positive = gain is not None and gain > 0
        passes = {
            "ordinary": eligible,
            "eligibility_only": eligible and every,
            "gain_only": eligible and (k == 1 or positive),
            "actual": eligible and every and (k == 1 or positive),
        }
        for name, yes in passes.items():
            if yes:
                allowed[name].append(c)
        result["candidates"].append(
            {
                "k": k,
                "full_status": c["status"],
                "fold_statuses": states,
                "fold_ineligible_count": sum(s != "eligible" for s in states),
                "pooled_gain_all_fitted_folds": gain,
                "fold_mean_gains": fold_gains,
                "minimum_unit_gain": min(gains) if gains else None,
                "maximum_unit_gain": max(gains) if gains else None,
                "negative_units": sum(x < 0 for x in gains),
                "passes": passes,
            }
        )
    for name, rows in allowed.items():
        if not rows:
            return {**result, "error": "Unavailable counterfactual baseline"}
        result[name + "_k"] = min(rows, key=lambda c: (c["aic"], c["k"]))["k"]
    result["status"] = "complete"
    return result
