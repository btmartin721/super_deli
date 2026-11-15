"""Small descriptive summaries and one frozen follow-up decision; no defaults selected."""

from collections import Counter

import numpy as np
from scipy.stats import beta


def interval(successes, n):
    if not n:
        return None
    return [
        float(beta.ppf(0.025, successes, n - successes + 1)) if successes else 0.0,
        float(beta.ppf(0.975, successes + 1, n - successes)) if successes < n else 1.0,
    ]


def summarize(main, audit):
    observations = main["records"]
    cells = []
    for origin in ("archived", "fresh"):
        for n in sorted({r.get("n") for r in observations if r["origin"] == origin and r.get("n")}):
            for law in sorted(
                {r["law"] for r in observations if r["origin"] == origin and r.get("n") == n}
            ):
                chosen = [
                    r
                    for r in observations
                    if r["origin"] == origin and r.get("n") == n and r["law"] == law
                ]
                names = list(dict.fromkeys(v["name"] for r in chosen for v in r["variants"]))
                for name in names:
                    rows = [
                        next(
                            (v for v in r["variants"] if v["name"] == name),
                            dict(status="unavailable"),
                        )
                        for r in chosen
                    ]
                    complete = [r for r in rows if r["status"] == "complete"]
                    failed = len(rows) - len(complete)
                    valid = [r for r in complete if r.get("correct_recovery") is not None]
                    control = [r for r in complete if r.get("control_split") is not None]
                    success = sum(bool(r["correct_recovery"]) for r in valid)
                    splits = sum(bool(r["control_split"]) for r in control)
                    values = [
                        r["shape_energy_retained"]
                        for r in complete
                        if r.get("shape_energy_retained") is not None
                    ]
                    cells.append(
                        dict(
                            origin=origin,
                            n=n,
                            law=law,
                            arm=name,
                            datasets=len(rows),
                            completed=len(complete),
                            failed=failed,
                            selected_k_counts=dict(Counter(r["selected_k"] for r in complete)),
                            correct_recovery_count=success if valid else None,
                            correct_recovery_interval=interval(success, len(valid))
                            if valid
                            else None,
                            control_split_count=splits if control else None,
                            control_split_interval=interval(splits, len(control))
                            if control
                            else None,
                            recovery_failure_bounds=[
                                success / len(rows),
                                (success + failed) / len(rows),
                            ]
                            if valid
                            else None,
                            median_shape_energy_retained=float(np.median(values))
                            if values
                            else None,
                        )
                    )
    source_shape = [r for r in audit["records"] if r["law"] == "shape_only"]
    low = sum(r["joint_view_centroid_energy_retained"]["outline"] < 0.5 for r in source_shape)

    def counts(law, arm, field):
        rows = [
            v
            for r in observations
            if r["origin"] == "fresh" and r["law"] == law
            for v in r["variants"]
            if v["name"] == arm
        ]
        return sum(bool(v.get(field, False)) for v in rows)

    shape_gain = counts("shape_only", "all_pc8_min9", "correct_recovery") - counts(
        "shape_only", "all_pc2_min9", "correct_recovery"
    )
    split_delta = counts("continuous_control", "all_pc8_min9", "control_split") - counts(
        "continuous_control", "all_pc2_min9", "control_split"
    )
    complementary_delta = counts("complementary", "all_pc8_min9", "correct_recovery") - counts(
        "complementary", "all_pc2_min9", "correct_recovery"
    )

    def failures(arm):
        return sum(
            v["status"] != "complete"
            for r in observations
            if r["origin"] == "fresh"
            for v in r["variants"]
            if v["name"] == arm
        )

    extra_failures = failures("all_pc8_min9") - failures("all_pc2_min9")
    full_energy = [
        v.get("shape_energy_retained")
        for r in observations
        for v in r["variants"]
        if v["name"] == "all_pc8_min9" and v["status"] == "complete"
    ]
    isometry = bool(full_energy) and all(abs(x - 1) < 1e-9 for x in full_energy)
    go = (
        low >= 3
        and isometry
        and shape_gain >= 2
        and split_delta <= 0
        and complementary_delta >= 0
        and extra_failures <= 0
    )
    decision = dict(
        status="go_for_bounded_opt_in_followup" if go else "no_go_for_immediate_implementation",
        archived_shape_cases_below_half_energy=low,
        archived_shape_cases=len(source_shape),
        full_retention_numerically_verified=isometry,
        fresh_shape_correct_recovery_gain=shape_gain,
        fresh_control_split_delta=split_delta,
        fresh_complementary_recovery_delta=complementary_delta,
        fresh_additional_failures=extra_failures,
        interpretation=(
            "Frozen descriptive research gate; no general size guarantee or automatic default"
        ),
    )
    return dict(
        schema_version=1,
        observations=len(observations),
        cells=cells,
        decision=decision,
        searches=main["planned_searches"],
        individual_records_published=0,
        uncertainty=(
            "Unadjusted exact 95% binomial intervals for tiny independent cells; "
            "descriptive, not simultaneous"
        ),
        limitations=[
            "Cached-score arms hold training-view transforms fixed; they are n"
            "ot full-pipeline calibration.",
            "BIC costs are compared within each score space; absolute BIC acro"
            "ss dimensions is not ranked.",
            "Native versus common-order head contrasts may differ through optimization row order.",
            "Controls/conflicting laws have no privileged four-species truth.",
            "In-sample linear recovery can saturate in high dimension; it is n"
            "ot a biological information bound.",
            "Four fresh datasets per cell support only a bounded descriptive decision.",
        ],
    )
