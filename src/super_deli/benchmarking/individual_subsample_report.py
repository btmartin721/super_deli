"""Small parent-level paired operating-range summaries; no automatic promotion."""

from collections import Counter, defaultdict

import numpy as np
from scipy.stats import beta, t


def exact_event(values):
    k = sum(values)
    n = len(values)
    return dict(
        successes=int(k),
        parents=n,
        rate=k / n,
        exact_95_interval=[
            0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)),
            1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k)),
        ],
    )


def descriptive(values, *, difference=False):
    a = np.asarray(values)
    m = float(a.mean())
    h = float(t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 1.0
    return dict(
        mean=m,
        independent_parents=len(a),
        descriptive_95_interval=[max(-1.0 if difference else 0.0, m - h), min(1.0, m + h)],
    )


def aggregate(summary):
    cells = defaultdict(list)
    for p in summary["parents"]:
        cells[(p["task"]["n"], p["task"]["law"])].append(p)
    result = []
    for (n, law), parents in sorted(cells.items()):
        rates = {}
        arms = {}
        for arm in ("ordinary", "subsample"):
            values = [p["arms"][arm][1 if arm == "ordinary" else 0 :] for p in parents]
            good = [[r for r in rr if r["status"] == "complete"] for rr in values]
            all_good = [r for rr in good for r in rr]
            split = [sum(r["selected_k"] > 1 for r in rr) / 25 for rr in good]
            recover = [
                sum(r["selected_k"] == 4 and r["ari"] >= 0.8 for r in rr) / 25 for rr in good
            ]
            rate = split if law == "continuous_control" else recover
            rates[arm] = rate
            arms[arm] = dict(
                requested=25 * len(parents),
                attempted=sum(map(len, values)),
                withheld=25 * len(parents) - len(all_good),
                selected_k=dict(Counter(str(r["selected_k"]) for r in all_good)),
                k4_status=dict(Counter(r["k4_status"] for r in all_good)),
                split_or_recovery_lower=descriptive(rate),
                split_or_recovery_upper=descriptive(
                    [x + (25 - len(g)) / 25 for x, g in zip(rate, good, strict=True)]
                ),
                correct_four_group_count=None
                if law == "continuous_control"
                else sum(r["selected_k"] == 4 and r["ari"] >= 0.8 for r in all_good),
                control_split_count=sum(r["selected_k"] > 1 for r in all_good)
                if law == "continuous_control"
                else None,
                median_original_ari=None
                if not all_good or law == "continuous_control"
                else float(np.median([r["ari"] for r in all_good])),
                mean_external_gain=None
                if not all_good
                else float(np.mean([r["evaluation_gain"] for r in all_good])),
                split_without_positive_external_gain=sum(
                    r["selected_k"] > 1 and r["evaluation_gain"] <= 0 for r in all_good
                ),
                any_failure_parent_event=exact_event(
                    [len(rr) < 25 or any(r["status"] != "complete" for r in rr) for rr in values]
                ),
            )
        point = [p["arms"]["ordinary"][0] for p in parents]
        result.append(
            dict(
                n=n,
                law=law,
                independent_parents=len(parents),
                point_selected_k=dict(Counter(str(r.get("selected_k", "withheld")) for r in point)),
                point_correct_or_split=exact_event(
                    [
                        r["status"] == "complete"
                        and (
                            r["selected_k"] > 1
                            if law == "continuous_control"
                            else r["selected_k"] == 4 and r["ari"] >= 0.8
                        )
                        for r in point
                    ]
                ),
                arms=arms,
                paired_subsample_minus_bootstrap=descriptive(
                    [a - b for a, b in zip(rates["subsample"], rates["ordinary"], strict=True)],
                    difference=True,
                ),
                mean_subsample_omitted_pair_coverage=float(
                    np.mean([p["subsample_summary"]["omitted_pair_coverage"] for p in parents])
                ),
                mean_subsample_included_pair_coverage=float(
                    np.mean([p["subsample_summary"]["included_pair_coverage"] for p in parents])
                ),
            )
        )
    return dict(
        passed=True,
        searches=summary["searches"],
        attempted_searches=summary["attempted_searches"],
        withheld=summary["withheld"],
        parent_cells=result,
        exact_replay_arrays=sum(r["exact_arrays"] for r in summary["replays"]),
        interpretation=(
            "Conditional perturbation measures, not interchangeable probabilities. "
            "Marginal descriptive intervals condition on fixed sampling/optimizer streams; "
            "draws repeat within independent parents. No species significance, "
            "general guarantee or automatic promotion."
        ),
    )
