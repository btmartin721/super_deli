"""Small parent-level summaries; draws are repeated conditional measurements."""

from collections import Counter, defaultdict

import numpy as np
from scipy.stats import t


def interval(values):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return None
    m = float(x.mean())
    h = float(t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 1.0
    return dict(
        mean=m,
        descriptive_95_interval=[max(0.0, m - h), min(1.0, m + h)],
        independent_parents=len(x),
    )


def aggregate(summary):
    cells = defaultdict(list)
    for p in summary["parents"]:
        cells[(p["head_only"], p["task"]["n"], p["task"]["law"])].append(p)
    output = []
    for (head, n, law), parents in sorted(cells.items()):
        arms = {}
        for arm in ("ordinary", "unique", "reference"):
            rows = [r for p in parents for r in p["arms"][arm][(1 if arm == "ordinary" else 0) :]]
            complete = [r for r in rows if r["status"] == "complete"]
            rates = []
            upper = []
            recovery = []
            prediction = []
            for p in parents:
                rr = p["arms"][arm][(1 if arm == "ordinary" else 0) :]
                ok = [r for r in rr if r["status"] == "complete"]
                rates.append(sum(r["selected_k"] > 1 for r in ok) / 25)
                upper.append((sum(r["selected_k"] > 1 for r in ok) + 25 - len(ok)) / 25)
                recovery.append(sum(r["selected_k"] == 4 and r["ari"] >= 0.8 for r in ok) / 25)
                prediction.append(sum(r["evaluation_gain"] > 0 for r in ok) / 25)
            arms[arm] = dict(
                requested=25 * len(parents),
                attempted=len(rows),
                unattempted=25 * len(parents) - len(rows),
                withheld=25 * len(parents) - len(complete),
                selected_k=dict(Counter(str(r["selected_k"]) for r in complete)),
                k4_status=dict(Counter(r["k4_status"] for r in complete)),
                split_lower=interval(rates),
                split_upper=interval(upper),
                correct_four_group_lower=interval(recovery) if law == "shape_only" else None,
                positive_external_gain_lower=interval(prediction),
                mean_external_gain=None
                if not complete
                else float(np.mean([r["evaluation_gain"] for r in complete])),
                median_original_ari=None
                if not complete or law != "shape_only"
                else float(np.median([r["ari"] for r in complete])),
                mean_common_point_ari=None
                if not any(r["point_ari"] is not None for r in complete)
                else float(
                    np.mean([r["point_ari"] for r in complete if r["point_ari"] is not None])
                ),
                oob_units=sum(r.get("oob_units", 0) for r in complete),
                reference_oob="undefined"
                if arm == "reference"
                else "same original-draw absence mask",
            )
        output.append(
            dict(head_only=head, n=n, law=law, independent_parents=len(parents), arms=arms)
        )
    return dict(
        passed=True,
        searches=summary["searches"],
        parent_cells=output,
        replay_arrays=sum(r["exact_arrays"] for r in summary["replays"]),
        interpretation=(
            "Descriptive parent-level intervals; no new p-values or resampling recommendation. "
            "Unique support changes weights, N, BIC and preprocessing. "
            "Head-only controls bypass ingestion/preprocessing."
        ),
    )
