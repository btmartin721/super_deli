"""Plot audited Gaussian score-test results without running numerical fits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def plot(directory, *, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    directory, destination = Path(directory), Path(output_dir)
    summary = json.loads((directory / "summary.json").read_text())
    with (directory / "datasets.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if any(rate["n_available"] == 0 for rate in summary["rates"]):
        raise ValueError("Plot requires at least one available p-value in every scenario")
    destination.mkdir(exist_ok=False)
    names = {
        "spherical_gaussian": "Spherical Gaussian (null)",
        "correlated_gaussian": "Correlated Gaussian (null)",
        "separated_mixture": "Separated mixture (power)",
        "heavy_tailed": "Heavy-tailed t3 (stress)",
    }
    colors = ("#0072B2", "#56B4E9", "#009E73", "#D55E00")
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7), constrained_layout=True)
    left, right = axes
    for i, (rate, color) in enumerate(zip(summary["rates"], colors, strict=True)):
        mean, interval = rate["mean"], rate["wilson_95"]
        name = names[rate["scenario"]]
        left.errorbar(
            mean,
            i,
            xerr=[[mean - interval[0]], [interval[1] - mean]],
            fmt="o",
            capsize=4,
            color=color,
            markersize=7,
        )
        left.text(
            min(mean + 0.06, 0.84),
            i + 0.2,
            f"{rate['events']}/{rate['n_available']} available",
            fontsize=9,
            color=color,
        )
        values = sorted(
            float(r["p_value"])
            for r in rows
            if r["scenario"] == rate["scenario"] and r.get("p_value")
        )
        right.step(
            [0, *values, 1],
            [0, *np.arange(1, len(values) + 1) / len(values), 1],
            where="post",
            label=name,
            color=color,
            linewidth=1.8,
        )
    alpha = summary["study"]["calibration"]["alpha"]
    left.axvline(alpha, color="0.5", linestyle="--", linewidth=1)
    left.set(
        yticks=range(4),
        yticklabels=list(names.values()),
        xlim=(-0.02, 1.02),
        ylim=(3.65, -0.5),
        xlabel=f"Rejection rate at alpha={alpha:g}",
        title="A  Independent-dataset rates and Wilson 95% intervals",
    )
    left.xaxis.set_major_formatter(PercentFormatter(1))
    right.plot([0, 1], [0, 1], linestyle="--", color="0.65", linewidth=1, label="Uniform reference")
    right.set(
        xlim=(0, 1),
        ylim=(0, 1.03),
        xlabel="Approximate p-value threshold",
        ylabel="Fraction of available datasets at or below threshold",
        title="B  Empirical p-value distributions",
    )
    right.yaxis.set_major_formatter(PercentFormatter(1))
    right.legend(loc="lower right", fontsize=8, frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.15)
    fig.suptitle(
        "Frozen score-space Gaussian calibration: null, power and stress checks", fontsize=13
    )
    for extension in ("pdf", "svg", "png"):
        path = destination / f"score-calibration.{extension}"
        fig.savefig(path, dpi=200)
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot(args.analysis, output_dir=args.output_dir)
