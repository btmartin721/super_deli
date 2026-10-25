"""Paired dataset effects, with no bootstrap-draw pseudoreplication."""

import argparse
import json
from pathlib import Path


def plot(summary, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    data = json.loads(Path(summary).read_text())
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    cells = [
        ("four_n64_s3_0", "64 localities; signal 3"),
        ("four_n64_s4_5", "64 localities; signal 4.5"),
        ("four_n256_s3_0", "256 localities; signal 3"),
    ]
    lookup = {(r["cell"], r["repetition"], r["arm"]): r for r in data["datasets"]}
    colors = ["#0072B2", "#D55E00"]
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.9), layout="constrained")
    for col, (cell, title) in enumerate(cells):
        reps = sorted({r["repetition"] for r in data["datasets"] if r["cell"] == cell})
        for row, (key, label) in enumerate(
            [
                ("k4", "Bootstrap K=4 frequency"),
                ("same_common_k", "Frequency matching legacy point K"),
            ]
        ):
            ax = axes[row, col]
            for i, rep in enumerate(reps):
                values = [
                    lookup[cell, rep, arm][key]["conditional_frequency"]
                    for arm in ("legacy", "scaled")
                ]
                if any(v is None for v in values):
                    continue
                # Small fixed offsets separate independently simulated datasets.
                offset = (i - (len(reps) - 1) / 2) * 0.018
                ax.plot([offset, 1 + offset], values, color="0.68", lw=1, zorder=1)
                for x, value, color in zip([offset, 1 + offset], values, colors, strict=True):
                    ax.scatter(x, value, color=color, s=31, zorder=2)
            ax.set(
                xticks=[0, 1],
                xticklabels=["Legacy", "Scaled folds"],
                xlim=(-0.2, 1.2),
                ylim=(-0.035, 1.035),
            )
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="0.9", lw=0.7)
            if row == 0:
                ax.set_title(title, fontsize=11)
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
    fig.suptitle("Paired whole-locality bootstrap selections", fontsize=14)
    fig.supxlabel(
        "Each connected pair is one independent dataset; 99 draws per policy.\n"
        "Five datasets per cell; descriptive effects, conditional on fixed folds.",
        fontsize=10,
    )
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(out / ("paired_recovery." + suffix), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot(args.summary, args.output_dir)


if __name__ == "__main__":
    main()
