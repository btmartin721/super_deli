"""Publish compact, audited diagnostic aggregates and a rate figure only."""

import argparse
import csv
import gzip
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def points(path):
    with gzip.open(path, "rt") as handle:
        return [json.loads(s) for s in handle]


def describe(values):
    import numpy as np

    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite mechanism evidence")
    if not len(x):
        return {"n": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "n": len(x),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "minimum": float(x.min()),
        "maximum": float(x.max()),
    }


def selected(row):
    return next(c for c in row["metrics"]["candidates"] if c["k"] == row["metrics"]["aic_k"])


def mechanisms(rows):
    complete = [r for r in rows if r["status"] == "complete"]
    multiple = [r for r in complete if r["metrics"]["aic_k"] > 1]
    metrics = [selected(r) for r in multiple]
    result = {
        "available": len(complete),
        "aic_multiple": len(multiple),
        "multiple_component_diagnostics": {},
    }
    for key in (
        "minimum_eigenvalue_over_regularizer",
        "mean_assignment_entropy",
        "count_label_mutual_information",
        "heldout_mean_logdensity_gain_over_one",
    ):
        result["multiple_component_diagnostics"][key] = describe(
            [c[key] for c in metrics if key in c]
        )
    if metrics and "heldout_mean_logdensity_gain_over_one" in metrics[0]:
        result["multiple_negative_heldout_gain"] = sum(
            c["heldout_mean_logdensity_gain_over_one"] < 0 for c in metrics
        )
    result["minimum_selected_hard_size"] = describe([min(c["hard_cluster_sizes"]) for c in metrics])
    result["maximum_component_mean_mahalanobis_squared"] = describe(
        [max(v for v in c["component_mean_mahalanobis_squared"] if v is not None) for c in metrics]
    )
    result["candidate_status_counts"] = {
        str(k): {
            status: sum(
                next(c for c in r["metrics"]["candidates"] if c["k"] == k)["status"] == status
                for r in complete
            )
            for status in ("eligible", "below_minimum_size", "nonconverged", "insufficient_samples")
        }
        for k in range(1, 5)
    }
    # Keep all actual statuses if a future additive audit uses another name.
    for k in range(1, 5):
        statuses = [
            next(c for c in r["metrics"]["candidates"] if c["k"] == k)["status"] for r in complete
        ]
        result["candidate_status_counts"][str(k)] = {
            s: statuses.count(s) for s in sorted(set(statuses))
        }
    return result


def make_figure(summary, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {
        "independent_gaussian": "Independent",
        "correlated_gaussian": "Correlated",
        "zero_conditional_gaussian": "Zero conditional",
    }
    arms = {
        "baseline": ("Baseline", "#0072B2"),
        "homogeneous_mean_noise": ("Equalized mean noise", "#009E73"),
        "starts_100": ("100 starts", "#D55E00"),
        "scores": ("Direct Gaussian scores", "#6F4E9C"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8), sharex=True)
    for ax, n in zip(axes, (64, 256), strict=True):
        cells = [r for r in summary["cells"] if r["n"] == n]
        labels = []
        for y, r in enumerate(cells):
            a = r["primary_aic_multiple_components"]
            label, color = arms[r["arm"]]
            labels.append(
                (names.get(r["case"], "Direct scores") + " / " + label)
                if r["arm"] != "scores"
                else label
            )
            if a["available"]:
                lo, hi = a["bonferroni_familywise_95"]
                x = a["rate_among_available"]
                ax.errorbar(x, y, xerr=[[x - lo], [hi - x]], fmt="o", color=color, capsize=3)
                ax.text(1.02, y, f"{a['events']}/{a['available']}", va="center", fontsize=8)
        ax.set_yticks(range(len(labels)), labels=labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(-0.02, 1.0)
        ax.set_title(f"{n} training units")
        ax.set_xlabel("Fraction selecting K > 1 by AIC")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(
        "Gaussian controls: component selection\n"
        "95% simultaneous exact intervals across 20 prespecified rates",
        fontsize=12,
    )
    fig.tight_layout()
    directory.mkdir(parents=True, exist_ok=False)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(directory / f"aic-selection-rates.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    summary = read(args.audit / "summary.json")
    fresh, old = points(args.audit / "points.jsonl.gz"), points(args.sources / "points.jsonl.gz")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "fresh": [],
        "archived": [],
        "optimization": [],
        "scope": (
            "Descriptive mechanisms; no refits, p-values, "
            "model selection on test data or default changes"
        ),
    }
    for row in summary["cells"]:
        group = [r for r in fresh if r["cell"] == row["cell"] and r["arm"] == row["arm"]]
        result["fresh"].append({"cell": row["cell"], "arm": row["arm"], **mechanisms(group)})
    for cohort, case, n in sorted({(r["cohort"], r["case"], r["n"]) for r in old}):
        group = [r for r in old if (r["cohort"], r["case"], r["n"]) == (cohort, case, n)]
        result["archived"].append({"cohort": cohort, "case": case, "n": n, **mechanisms(group)})
    for cell in sorted({r["cell"] for r in fresh if r["arm"] == "baseline"}):
        by = {(r["repetition"], r["arm"]): r for r in fresh if r["cell"] == cell}
        comparisons = []
        for i in sorted({r[0] for r in by}):
            a, b = by[i, "baseline"], by[i, "starts_100"]
            if a["status"] == b["status"] == "complete":
                comparisons.append((a, b))
        detail = {
            "cell": cell,
            "available_pairs": len(comparisons),
            "selected_k_changes": sum(a["aic_k"] != b["aic_k"] for a, b in comparisons),
            "candidate_comparisons": [],
        }
        for k in range(1, 5):
            pairs = [
                (
                    next(c for c in a["metrics"]["candidates"] if c["k"] == k),
                    next(c for c in b["metrics"]["candidates"] if c["k"] == k),
                )
                for a, b in comparisons
            ]
            pairs = [
                (a, b)
                for a, b in pairs
                if a["log_likelihood"] is not None and b["log_likelihood"] is not None
            ]
            detail["candidate_comparisons"].append(
                {
                    "k": k,
                    "long_minus_baseline_loglikelihood": describe(
                        [b["log_likelihood"] - a["log_likelihood"] for a, b in pairs]
                    ),
                    "eligibility_changes": sum(
                        (a["status"] == "eligible") != (b["status"] == "eligible") for a, b in pairs
                    ),
                }
            )
        result["optimization"].append(detail)
    for name, value in (
        ("summary.json", summary),
        ("mechanisms.json", result),
        ("archived-summary.json", read(args.sources / "summary.json")),
    ):
        with (args.output_dir / name).open("x") as h:
            json.dump(value, h, indent=2, sort_keys=True)
            h.write("\n")
    with (args.output_dir / "rates.csv").open("x", newline="") as h:
        writer = csv.writer(h)
        writer.writerow(
            [
                "case",
                "n",
                "arm",
                "events",
                "available",
                "requested",
                "rate",
                "pointwise_lo",
                "pointwise_hi",
                "simultaneous_lo",
                "simultaneous_hi",
            ]
        )
        for r in summary["cells"]:
            a = r["primary_aic_multiple_components"]
            writer.writerow(
                [
                    r["case"],
                    r["n"],
                    r["arm"],
                    a["events"],
                    a["available"],
                    a["requested"],
                    a["rate_among_available"],
                    *(a["exact_interval_95"] or [None, None]),
                    *(a["bonferroni_familywise_95"] or [None, None]),
                ]
            )
    make_figure(summary, args.figures)
    print(
        json.dumps(
            {
                "published_cells": len(summary["cells"]),
                "private_fresh_points": len(fresh),
                "private_archived_points": len(old),
            }
        )
    )


if __name__ == "__main__":
    main()
