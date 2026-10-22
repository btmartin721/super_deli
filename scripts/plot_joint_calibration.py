"""Plot audited joint-null rejection rates without changing scientific evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NAMES = {
    "zero_association": "Zero association · null",
    "linear_association": "Linear association · null",
    "complementary": "Complementary four groups · alternative",
    "heavy_tailed": "Heavy-tailed residuals · stress",
    "cline": "Continuous cline · stress",
}


def plot(results, output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    summary = json.loads((results / "summary.json").read_text())
    rates = {row["case"]: row for row in summary["rates"]}
    output.mkdir(exist_ok=False)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    figure, axis = plt.subplots(figsize=(9, 4.8))
    for i, (case, color) in enumerate(
        zip(NAMES, ("#0072B2", "#0072B2", "#009E73", "#D55E00", "#D55E00"), strict=True)
    ):
        rate = rates[case]
        if rate["mean"] is None:
            continue
        low, high = rate["wilson_95"]
        axis.errorbar(
            rate["mean"],
            i,
            xerr=[[rate["mean"] - low], [high - rate["mean"]]],
            fmt="o",
            color=color,
            capsize=4,
            markersize=7,
        )
        axis.annotate(
            f"{rate['events']}/{rate['n_available']}",
            (rate["mean"], i),
            xytext=(-6 if rate["mean"] > 0.9 else 6, 9),
            textcoords="offset points",
            ha="right" if rate["mean"] > 0.9 else "left",
        )
    axis.axvline(0.05, color="#777777", linestyle="--", linewidth=1, label="Declared alpha = 5%")
    axis.set(
        title="Joint-law departures across five generating scenarios",
        xlabel="Rejected / available independent datasets",
        xlim=(-0.02, 1.02),
        ylim=(4.55, -0.65),
        yticks=list(range(5)),
        yticklabels=list(NAMES.values()),
    )
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower right", frameon=False)
    figure.text(
        0.02,
        0.025,
        "Pointwise Wilson 95% intervals; 25 datasets/scenario. "
        "Stress rejection measures misspecification sensitivity.\n"
        "Approximate fitted-null tests; no general size guarantee "
        "or species-boundary significance.",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.1, 1, 1))
    for extension in ("png", "pdf", "svg"):
        path = output / f"joint-calibration.{extension}"
        figure.savefig(path, dpi=180)
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    plot(args.results, args.output_dir)
