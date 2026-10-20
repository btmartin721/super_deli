"""Render audited study summaries as vector and raster figures without refitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot_summary(summary, *, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    destination = Path(output_dir)
    destination.mkdir(exist_ok=False)
    groups = summary["groups"]
    scenarios = list(dict.fromkeys(g["scenario"] for g in groups))
    arms = list(dict.fromkeys(g["arm"] for g in groups))
    labels = {
        "null_matched": "No group structure",
        "strong_matched": "Strong shared structure",
        "weak_matched": "Weak shared structure",
        "complementary_matched": "Complementary signals",
        "conflicting_matched": "Ecological conflict",
        "missing_cells": "Missing cells",
        "missing_views": "Missing ecology records",
        "unequal_grouped": "Unequal grouped sampling",
    }
    names = {
        "full": "All four views",
        "genome_environment": "Genomic + ecological",
        "traits_shape": "Traits + shape",
    }
    colors, markers = ("#0072B2", "#D55E00", "#009E73", "#CC79A7"), ("o", "^", "s", "D")
    plt.rcParams.update(
        {
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 10.5), constrained_layout=True)
    panels = (
        ("ari", "A  Partition recovery", "Adjusted Rand index", (-0.05, 1.05)),
        ("k_correct", "B  K recovery among completed fits", "Exact K recovery", (0, 1.05)),
        (
            "coassignment_squared_error",
            "C  Bootstrap recovery error",
            "Mean squared coassignment error",
            (-0.02, 1.02),
        ),
        (
            "oob_pair_coverage",
            "D  OOB pair coverage",
            "Fraction with an unaffected OOB estimate",
            (0, 1.05),
        ),
    )
    for axis, (metric, title, xlabel, limits) in zip(axes.flat, panels, strict=True):
        for a, arm in enumerate(arms):
            rows = [
                next(g for g in groups if g["scenario"] == s and g["arm"] == arm) for s in scenarios
            ]
            y = [
                len(scenarios) - 1 - i + (a - (len(arms) - 1) / 2) * 0.2
                for i in range(len(scenarios))
            ]
            for position, group in zip(y, rows, strict=True):
                values = group["metrics"][metric]
                if values["mean"] is None:
                    continue
                point = values["mean"]
                if metric == "k_correct":
                    low, high = group["rates"][metric]["wilson_95"]
                    error = [[max(0, point - low)], [max(0, high - point)]]
                else:
                    error = values["mcse"] or 0
                axis.errorbar(
                    point,
                    position,
                    xerr=error,
                    fmt=markers[a % len(markers)],
                    color=colors[a % len(colors)],
                    markersize=5,
                    capsize=2,
                    linewidth=1,
                )
            axis.plot(
                [],
                [],
                marker=markers[a % len(markers)],
                linestyle="none",
                color=colors[a % len(colors)],
                label=names.get(arm, arm.replace("_", " ")),
            )
        axis.set_yticks(range(len(scenarios)), [labels.get(s, s) for s in reversed(scenarios)])
        axis.set_ylim(-0.65, len(scenarios) - 0.35)
        axis.set_xlim(*limits)
        axis.set_title(title, loc="left", fontsize=12, weight="bold", pad=10)
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", color="0.9", linewidth=0.6)
        axis.set_axisbelow(True)
        if metric in {"k_correct", "oob_pair_coverage"}:
            axis.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    axes[0, 0].legend(loc="lower left", frameon=True, fontsize=9)
    figure.suptitle(
        "Integrative clustering: recovery and conditional bootstrap behavior",
        fontsize=15,
        weight="bold",
    )
    figure.supxlabel(
        "Points: dataset means. Bars: ±MCSE; K recovery uses pointwise 95% Wilson "
        "Monte Carlo intervals.\nAvailability and failure denominators are reported in the "
        "tables; these are not species-support intervals.",
        fontsize=9,
    )
    for suffix in ("pdf", "svg", "png"):
        target = destination / f"integrative-study.{suffix}"
        figure.savefig(target, dpi=200)
        if suffix == "svg":
            target.write_text(
                "\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n"
            )
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    plot_summary(json.loads(args.summary.read_text()), output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
