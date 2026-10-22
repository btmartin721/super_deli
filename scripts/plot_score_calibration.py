"""Plot audited score-space or raw-reference tests without running numerical fits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def plot(
    directory,
    *,
    output_dir,
    reference_pipeline=False,
    empirical_pipeline=False,
    grouped_pipeline=False,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    directory, destination = Path(directory), Path(output_dir)
    summary = json.loads((directory / "summary.json").read_text())
    with (directory / "datasets.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if sum((reference_pipeline, empirical_pipeline, grouped_pipeline)) > 1:
        raise ValueError("Select one plot profile")
    if not grouped_pipeline and any(rate["n_available"] == 0 for rate in summary["rates"]):
        raise ValueError("Plot requires at least one available p-value in every scenario")
    destination.mkdir(exist_ok=False)
    names = {
        "spherical_gaussian": "Spherical Gaussian (null)",
        "correlated_gaussian": "Correlated Gaussian (null)",
        "separated_mixture": "Separated mixture (power)",
        "heavy_tailed": "Heavy-tailed t3 (stress)",
    }
    key = "scenario"
    title = "Frozen score-space Gaussian calibration: null, power and stress checks"
    label = "Approximate p-value threshold"
    filename = "score-calibration"
    if reference_pipeline:
        names = {
            "matched_null": "Matched individuals (null)",
            "matched_complementary": "Matched individuals (power)",
            "grouped_null": "Unequal grouped samples (null)",
            "grouped_complementary": "Unequal grouped samples (power)",
        }
        key = "case"
        title = "Full-pipeline reference-population calibration: null and power checks"
        label = "Reference-population p-value threshold"
        filename = "pipeline-calibration"
    if empirical_pipeline:
        if reference_pipeline:
            raise ValueError("Select one plot profile")
        names = {
            "spherical_gaussian": "Spherical raw Gaussian (null)",
            "correlated_gaussian": "Correlated raw Gaussian (null)",
            "complementary": "Complementary raw views (power)",
            "heavy_tailed": "Heavy-tailed raw t3 (stress)",
        }
        key = "case"
        title = "Fitted Gaussian null in raw continuous views: null, power and stress checks"
        label = "Approximate fitted raw-Gaussian p-value threshold"
        filename = "empirical-calibration"
    if grouped_pipeline:
        names = {
            "independent_gaussian": "Independent group Gaussian (null)",
            "correlated_gaussian": "Correlated group Gaussian (null)",
            "complementary": "Complementary grouped views (power)",
            "heavy_tailed": "Heavy-tailed grouped t3 (stress)",
        }
        key = "case"
        title = "Unequal grouped sampling: fitted hierarchical Gaussian calibration"
        label = "Approximate grouped fitted-null p-value threshold"
        filename = "grouped-calibration"
    colors = ("#0072B2", "#56B4E9", "#009E73", "#D55E00")
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7), constrained_layout=True)
    left, right = axes
    for i, (rate, color) in enumerate(zip(summary["rates"], colors, strict=True)):
        mean, interval = rate["mean"], rate["wilson_95"]
        name = names[rate[key]]
        if rate["n_available"] == 0:
            left.text(0.04, i, "No available tests", color=color, fontsize=9)
            continue
        left.errorbar(
            mean,
            i,
            xerr=[[mean - interval[0]], [interval[1] - mean]],
            fmt="o",
            capsize=4,
            color=color,
            markersize=7,
        )
        annotation = f"{rate['events']}/{rate['n_available']} rejected"
        if grouped_pipeline:
            annotation += f"; {rate['n_available']}/{summary['study']['n_datasets']} available"
        left.text(
            min(mean + 0.06, 0.5 if grouped_pipeline else 0.84),
            i + 0.2,
            annotation,
            fontsize=9,
            color=color,
        )
        values = sorted(
            float(r["p_value"]) for r in rows if r[key] == rate[key] and r.get("p_value")
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
        title=(
            "A  Available-test rates and Wilson 95% intervals"
            if grouped_pipeline
            else "A  Independent-dataset rates and Wilson 95% intervals"
        ),
    )
    left.xaxis.set_major_formatter(PercentFormatter(1))
    right.plot([0, 1], [0, 1], linestyle="--", color="0.65", linewidth=1, label="Uniform reference")
    right.set(
        xlim=(0, 1),
        ylim=(0, 1.03),
        xlabel=label,
        ylabel="Fraction of available datasets at or below threshold",
        title="B  Empirical p-value distributions",
    )
    right.yaxis.set_major_formatter(PercentFormatter(1))
    legend_options = (
        {"loc": "center right", "bbox_to_anchor": (0.99, 0.35)}
        if reference_pipeline or empirical_pipeline
        else {"loc": "lower right"}
    )
    if empirical_pipeline or grouped_pipeline:
        legend_options = {"loc": "upper left", "bbox_to_anchor": (0.02, 0.92)}
    right.legend(**legend_options, fontsize=8, frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.15)
    fig.suptitle(title, fontsize=13)
    for extension in ("pdf", "svg", "png"):
        path = destination / f"{filename}.{extension}"
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
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--reference-pipeline", action="store_true")
    profile.add_argument("--empirical-pipeline", action="store_true")
    profile.add_argument("--grouped-pipeline", action="store_true")
    args = parser.parse_args()
    plot(
        args.analysis,
        output_dir=args.output_dir,
        reference_pipeline=args.reference_pipeline,
        empirical_pipeline=args.empirical_pipeline,
        grouped_pipeline=args.grouped_pipeline,
    )
