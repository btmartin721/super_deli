"""Plot audited boundary calibration and paired test availability, without fits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def plot(analysis, comparison, *, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    analysis, comparison, destination = map(Path, (analysis, comparison, output_dir))
    summary = json.loads((analysis / "summary.json").read_text())
    with (comparison / "datasets.csv").open(newline="", encoding="utf-8") as handle:
        paired = list(csv.DictReader(handle))
    names = {
        "independent_gaussian": "Independent Gaussian (null)",
        "correlated_gaussian": "Correlated Gaussian (null)",
        "complementary": "Complementary groups (alternative)",
        "heavy_tailed": "Heavy-tailed t3 (stress)",
        "zero_conditional_gaussian": "Zero conditional covariance (null)",
        "rank_two_gaussian": "Rank-two conditional covariance (null)",
        "weak_conditional_gaussian": "Weak full-rank covariance (null)",
    }
    destination.mkdir(exist_ok=False)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.8, 6.2), constrained_layout=True)
    colors = ("#0072B2", "#56B4E9", "#009E73", "#D55E00", "#CC79A7", "#332288", "#117733")
    for i, (rate, color) in enumerate(zip(summary["rates"], colors, strict=True)):
        if rate["n_available"]:
            value, (lo, hi) = rate["mean"], rate["wilson_95"]
            left.errorbar(
                value, i, xerr=[[value - lo], [hi - value]], fmt="o", capsize=4, color=color
            )
            left.text(
                min(value + 0.03, 0.6),
                i + 0.2,
                f"{rate['events']}/{rate['n_available']} rejected; "
                f"{rate['n_available']}/25 available",
                fontsize=8.5,
                color=color,
            )
        else:
            left.text(0.03, i, "No available tests", fontsize=9, color=color)
    left.axvline(summary["study"]["calibration"]["alpha"], linestyle="--", color="0.5", linewidth=1)
    left.set(
        yticks=range(7),
        yticklabels=list(names.values()),
        ylim=(6.7, -0.5),
        xlim=(-0.02, 1.02),
        xlabel="Rejection proportion among available tests",
        title="A  Rates and pointwise Wilson 95% intervals",
    )
    cases = list(names)[:4]
    for offset, key, color, label in (
        (-0.18, "legacy_available", "#999999", "Legacy interior-only"),
        (0.18, "current_available", "#0072B2", "Conditional PSD"),
    ):
        counts = [sum(r[key] == "True" for r in paired if r["case"] == case) for case in cases]
        positions = np.arange(4) + offset
        right.barh(positions, np.array(counts) / 25, height=0.32, color=color, label=label)
        for value, y in zip(counts, positions, strict=True):
            right.text(value / 25 + 0.02, y, f"{value}/25", va="center", fontsize=9)
    right.set(
        yticks=range(4),
        yticklabels=[names[c] for c in cases],
        ylim=(3.6, -0.6),
        xlim=(0, 1.2),
        xticks=[0, 0.25, 0.5, 0.75, 1],
        xlabel="Available tests on paired original datasets",
        title="B  Availability on original cohorts",
    )
    right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=9)
    for axis in (left, right):
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", alpha=0.15)
    fig.suptitle(
        "Grouped fitted-null calibration: covariance boundaries and convergence", fontsize=13
    )
    for extension in ("pdf", "svg", "png"):
        path = destination / f"grouped-boundaries.{extension}"
        fig.savefig(path, dpi=200)
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot(args.analysis, args.comparison, output_dir=args.output_dir)
