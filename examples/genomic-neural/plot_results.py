"""Render the public aggregate only; no access to individual study records."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(source, output):
    data = json.loads(Path(source).read_text())
    arms = ["pca_diagonal", "pca_shared", "neural_seed_0", "neural_seed_1", "neural_seed_2"]
    names = ["PCA\ndiagonal", "PCA\nshared", "Neural\nseed 0", "Neural\nseed 1", "Neural\nseed 2"]
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharey=True)
    laws = [
        ("common_hwe", "Common-frequency single population: split rate"),
        ("rare_complete_hwe", "Rare-frequency single population: split rate"),
        ("four_groups", "Four populations: correct-K recovery rate"),
    ]
    for row, (law, title) in enumerate(laws):
        for col, n in enumerate((64, 128)):
            ax = axes[row, col]
            values = [
                next(c for c in data["cells"] if (c["law"], c["n"], c["arm"]) == (law, n, a))
                for a in arms
            ]
            p = np.array([c["event_count"] / 3 for c in values])
            intervals = np.array(
                [c["pointwise_exact_95_interval_with_failure_bounds"] for c in values]
            )
            ax.bar(
                range(5),
                p,
                color=["#236b8e", "#6baed6", "#d95f02", "#e6a04b", "#f2c078"],
                width=0.65,
            )
            ax.errorbar(
                range(5),
                p,
                yerr=np.stack((p - intervals[:, 0], intervals[:, 1] - p)),
                fmt="none",
                color="#333333",
                capsize=3,
            )
            ax.set(
                ylim=(-0.04, 1.08),
                xticks=range(5),
                xticklabels=names,
                ylabel="Fraction of 3 datasets",
            )
            ax.set_title(f"{title}\nN={n}; three independent datasets", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
    fig.suptitle(
        "Genomic neural baseline: strong-signal recovery does not prevent control splits",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.012,
        "Bars retain each fixed seed. Error bars: pointwise exact 95% intervals; "
        "seeds share datasets. No significance test.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(out.with_suffix(suffix), dpi=150)
    svg = out.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.source, args.output)
