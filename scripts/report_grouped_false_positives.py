"""Plot audited Gaussian null rejection rates without refitting any model."""

import argparse
import json
from pathlib import Path


def report(summary, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = (
        "independent_gaussian",
        "correlated_gaussian",
        "zero_conditional_gaussian",
        "rank_two_gaussian",
        "weak_conditional_gaussian",
    )
    labels = ("Independent", "Correlated", "Zero conditional", "Rank two", "Weak conditional")
    destination.mkdir(exist_ok=False)
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, constrained_layout=True
    )
    colors = {"fitted": "#CC6611", "known": "#0077BB"}
    for arm, offset in [("fitted", -0.12), ("known", 0.12)]:
        rows = {r["case"]: r for r in summary["primary_rates"] if r["arm"] == arm}
        for i, case in enumerate(names):
            r = rows[case]
            if r["rate_among_available"] is None:
                top.text(i + offset, 0, "withheld", rotation=90, ha="center")
                continue
            value = 100 * r["rate_among_available"]
            lo, hi = np.array(r["exact_interval_95"]) * 100
            a, b = np.array(r["bonferroni_familywise_95"]) * 100
            top.plot([i + offset] * 2, [a, b], color=colors[arm], alpha=0.3, lw=5)
            top.errorbar(
                i + offset,
                value,
                yerr=[[value - lo], [hi - value]],
                fmt="o",
                capsize=4,
                color=colors[arm],
                label=arm.capitalize() if i == 0 else None,
            )
    top.axhline(5, color="#555555", ls="--", lw=1, label="Nominal 5%")
    top.set_ylabel("Rejection rate (%)")
    top.set_title("Grouped likelihood-gain calibration: fixed-procedure Gaussian null validation")
    top.legend(frameon=False, ncol=3)
    for r in summary["paired_decisions"]:
        pos = names.index(r["case"])
        if r["known_minus_fitted"] is None:
            continue
        value = 100 * r["known_minus_fitted"]
        lo, hi = np.array(r["conservative_exact_difference_interval"]) * 100
        bottom.errorbar(
            pos, value, yerr=[[value - lo], [hi - value]], fmt="o", capsize=4, color="#228833"
        )
    bottom.axhline(0, color="#555555", ls="--", lw=1)
    bottom.set_ylabel("Known minus fitted\n(percentage points)")
    bottom.set_xticks(range(5), labels)
    bottom.set_title("Paired decisions; conservative exact 95% intervals")
    figure.supxlabel(
        "Rate bars: pointwise exact 95%; pale bars: Bonferroni across ten rates.\n"
        "Known parameters are simulation controls. No species-boundary interpretation.",
        fontsize=9,
    )
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(destination / f"grouped-false-positives.{suffix}", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report(json.loads(args.summary.read_text()), args.output_dir)


if __name__ == "__main__":
    main()
