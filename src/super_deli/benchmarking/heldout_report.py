"""Independent-parent operating ranges and prespecified descriptive stop decision."""

from collections import defaultdict

import numpy as np
from scipy.stats import beta


def event_rate(values):
    n = len(values)
    yes = sum(v is True for v in values)
    unknown = sum(v is None for v in values)

    def interval(alpha):
        return [
            0.0 if yes == 0 else float(beta.ppf(alpha / 2, yes, n - yes + 1)),
            1.0
            if yes + unknown == n
            else float(beta.ppf(1 - alpha / 2, yes + unknown + 1, n - yes - unknown)),
        ]

    return dict(
        parents=n,
        known_successes=yes,
        unavailable=unknown,
        identification_lower=yes / n,
        identification_upper=(yes + unknown) / n,
        marginal_exact_95_interval=interval(0.05),
        simultaneous_40_rate_95_interval=interval(0.05 / 40),
    )


def aggregate(summary):
    by_cell = defaultdict(list)
    for p in summary["parents"]:
        by_cell[p["task"]["cell"]].append(p)
    cells = []
    checks = []
    for cell, parents in sorted(by_cell.items()):
        task = parents[0]["task"]
        rates = {
            key: event_rate([p["primary"][key] for p in parents])
            for key in ("qualified", "positive", "agreement", "joint")
        }
        coverage = rates["qualified"]["identification_lower"] >= 0.9
        agree = rates["agreement"]["identification_lower"] >= 0.8
        target = task["design"] in {"balanced", "proportional"}
        flag = (
            rates["positive"]["identification_upper"] <= 0.1
            if task["design"] == "control"
            else rates["positive"]["identification_lower"] >= 0.8
            if target
            else None
        )
        checks.extend([coverage, agree])
        if flag is not None:
            checks.append(flag)
        point = [p["point"] for p in parents if p.get("point") is not None]
        means = [
            p["predictive"]["covered_individual_mean"]
            for p in parents
            if p.get("predictive") and p["predictive"]["covered_individual_mean"] is not None
        ]
        cells.append(
            dict(
                cell=cell,
                n=task["n"],
                design=task["design"],
                counts=task["counts"],
                parents=len(parents),
                rates=rates,
                point_correct=sum(p["correct"] for p in point),
                point_available=len(point),
                point_selected_k={
                    str(k): sum(p["selected_k"] == k for p in point) for k in range(1, 5)
                },
                median_point_ari=float(np.median([p["ari"] for p in point])) if point else None,
                median_rare_jaccard=float(np.median([p["rare_jaccard"] for p in point]))
                if point
                else None,
                median_internal_gain=float(np.median(means)) if means else None,
                descriptive_targets=dict(
                    coverage_passed=coverage, agreement_passed=agree, positive_flag_passed=flag
                ),
            )
        )
    passed = all(checks)
    return dict(
        passed=True,
        parents=len(summary["parents"]),
        planned_searches=summary["planned_searches"],
        native_searches=summary["native_searches"],
        baseline_searches=summary["baseline_searches"],
        attempted_searches=summary["attempted_searches"],
        cells=cells,
        descriptive_go=passed,
        decision="Experimental diagnostic only; no filter or default promotion"
        if passed
        else "No-go for a selection filter; retain transparent values and close this tuning branch",
        primary_rate_family=40,
        confidence_method="Exact intervals with unknown-event envelopes; Bonferroni family of40",
        interpretation=(
            "Independent parents supply uncertainty; repeated omitted "
            "individuals/draws are not replicates"
        ),
        p_value=None,
        new_selection_rule=False,
    )
