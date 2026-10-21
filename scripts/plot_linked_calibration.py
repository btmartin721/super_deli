"""Plot the audited genomic study without fitting models or changing evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

NAMES = {
    "common_hwe": "Common-frequency HWE",
    "linked_stress": "Perfect-LD blocks",
    "moderate_ld": "Moderate-LD blocks",
    "four_groups": "Constructed four groups",
    "cline_stress": "Continuous-cline stress",
    "split_block_stress": "Split-block stress",
}


def plot(results, output):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    summary = json.loads((results / "summary.json").read_text())
    with (results / "diagnostics.csv").open(newline="") as handle:
        diagnostics = {r["case"]: r for r in csv.DictReader(handle)}
    output.mkdir(exist_ok=False)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), constrained_layout=True)
    rates = sorted(summary["rates"], key=lambda r: list(NAMES).index(r["case"]))
    colors = ["#0072B2", "#0072B2", "#0072B2", "#009E73", "#D55E00", "#D55E00"]
    positions = np.arange(len(rates))
    for i, rate in enumerate(rates):
        if rate["mean"] is None:
            continue
        lo, hi = rate["wilson_95"]
        axes[0].errorbar(
            rate["mean"],
            i,
            xerr=[[rate["mean"] - lo], [hi - rate["mean"]]],
            fmt="o",
            color=colors[i],
            capsize=4,
            markersize=7,
        )
        axes[0].annotate(
            f"{rate['events']}/{rate['n_available']}",
            (rate["mean"], i),
            xytext=(-5 if rate["mean"] > 0.9 else 5, 9),
            textcoords="offset points",
            ha="right" if rate["mean"] > 0.9 else "left",
            fontsize=9,
        )
    axes[0].axvline(0.05, color="#777777", linestyle="--", linewidth=1, label="Declared alpha = 5%")
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    axes[0].set_title("A  Rejection and pointwise Wilson 95% intervals", loc="left", fontsize=11)
    axes[0].set_yticks(positions, [NAMES[r["case"]] for r in rates])
    axes[0].set_xlabel("Rejected / available independent datasets")
    left = np.zeros(len(rates))
    for k, color in enumerate(["#999999", "#56B4E9", "#009E73", "#E69F00"], 1):
        counts = np.array([int(diagnostics[r["case"]][f"point_k{k}"]) for r in rates])
        denominators = np.array([r["n_available"] for r in rates])
        fraction = np.divide(counts, denominators, out=np.zeros(len(rates)), where=denominators > 0)
        axes[1].barh(positions, fraction, left=left, color=color, height=0.55, label=f"K={k}")
        for i, count in enumerate(counts):
            if fraction[i] >= 0.1:
                axes[1].text(
                    left[i] + fraction[i] / 2, i, str(count), ha="center", va="center", fontsize=9
                )
        left += fraction
    axes[1].set_title("B  Selected K among available tests", loc="left", fontsize=11)
    axes[1].set_yticks(positions, [])
    axes[1].set_xlabel("Fraction of available datasets (labels are counts)")
    axes[1].legend(loc="lower left", bbox_to_anchor=(0, -0.25), ncol=4, frameon=False)
    for ax in axes:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(len(rates) - 0.45, -0.65)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle("Empirical block-null calibration under six generating laws", fontsize=13)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output / f"linked-calibration.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot(args.results, args.output_dir)
