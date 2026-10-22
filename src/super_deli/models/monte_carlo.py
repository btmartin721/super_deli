"""Shared upper-tail Monte Carlo accounting; assumptions belong to each null model."""

from __future__ import annotations

import numpy as np
from scipy.stats import beta


def summarize_upper_tail(
    observed,
    records,
    *,
    alpha,
    p_value_kind="Approximate fitted-Gaussian parametric bootstrap with plus-one rank",
    decision_key="reject_single_gaussian",
    interval_scope=None,
):
    """Keep failed draws in denominators; intervals describe numerical tail error only."""
    if not records or not np.isfinite(observed) or observed < 0:
        raise ValueError("A finite nonnegative statistic and simulation records are required")
    failed = sum(r["status"] == "failed" for r in records)
    if any(r["status"] not in {"complete", "failed"} for r in records):
        raise ValueError("Unknown null-draw status")
    statistics = [r["statistic"] for r in records if r["status"] == "complete"]
    if any(not np.isfinite(t) or t < 0 for t in statistics):
        raise ValueError("Null statistics must be finite and nonnegative")
    # Include ties (especially the atom at zero when K=1 wins) in the upper tail.
    exceedances = sum(t >= observed for t in statistics)
    requested = len(records)
    p = (1 + exceedances) / (requested + 1) if not failed else None
    interval = None
    if not failed:
        b, m = exceedances, requested
        interval = [
            0.0 if b == 0 else float(beta.ppf(0.025, b, m - b + 1)),
            1.0 if b == m else float(beta.ppf(0.975, b + 1, m - b)),
        ]
    return {
        "status": "available" if not failed else "incomplete",
        "requested": requested,
        "successful": requested - failed,
        "failed": failed,
        "exceedances": exceedances,
        "tail_rule": "T_null >= T_observed (ties included)",
        "p_value": p,
        "p_value_kind": p_value_kind,
        "minimum_p_value": 1 / (requested + 1),
        "alpha": alpha,
        "resolution_reaches_alpha": 1 / (requested + 1) <= alpha,
        decision_key: p <= alpha if p is not None else None,
        "failed_outcome_bounds": [
            (1 + exceedances) / (requested + 1),
            (1 + exceedances + failed) / (requested + 1),
        ],
        "conditional_null_tail_interval_95": interval,
        "interval_scope": interval_scope
        or (
            "Clopper-Pearson binomial interval for the tail probability conditional on "
            "the observed statistic and fitted null, not a species confidence interval; "
            "failed-outcome bounds identify unknown null statistics and are not intervals"
        ),
    }
