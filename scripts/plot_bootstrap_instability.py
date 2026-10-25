"""Static figure of the frozen paired diagnostic; no fitting or configuration selection."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import argparse

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="Plot descriptive paired bootstrap diagnostics")
parser.add_argument("summary", type=Path)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
d = json.loads(args.summary.read_text())
out = args.output_dir
out.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
names = ["four_n64_s3_0", "four_n64_s4_5", "four_n256_s3_0"]
titles = ["N=64, signal 3", "N=64, signal 4.5", "N=256, signal 3"]
colors = ["#1B6A80", "#D17C23", "#8A538C"]
for ax, cell, title in zip(axes, names, titles, strict=True):
    for j, arm in enumerate(["baseline", "unique", "extended"]):
        values = [
            v["k4"]["conditional_rate"]
            for v in d["datasets"]
            if v["cohort"] == "fresh"
            and v["cell"] == cell
            and v["arm"] == arm
            and v["draw_subset"] == "first_five"
        ]
        ax.scatter(
            np.arange(len(values)) * 0.014 + j - 0.063, values, s=30, color=colors[j], alpha=0.75
        )
        ax.plot([j - 0.18, j + 0.18], [np.mean(values)] * 2, color=colors[j], lw=3)
    ax.set(
        title=title,
        xticks=range(3),
        xticklabels=["Baseline", "Unique\nsupport", "100 starts"],
        ylim=(-0.04, 1.04),
    )
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Fraction selecting K=4 in first five draws")
fig.suptitle("Paired first-five-draw diagnostic controls", y=0.99)
fig.text(
    0.5,
    0.015,
    (
        "Dots mark the same ten datasets across paired arms; lines show means. "
        "Unique support changes weights, sample size, maps and order.\n"
        "Descriptive contrasts only: no replacement stability estimates or species support."
    ),
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=[0, 0.1, 1, 0.95])
for ext in ["png", "svg", "pdf"]:
    fig.savefig(out / f"paired-recovery.{ext}", dpi=180)
print("saved three paired-recovery figures")
