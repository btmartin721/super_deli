"""Plot audited small aggregates; no fitting, tuning or hidden source data."""

import argparse
import json
from pathlib import Path


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = json.loads(Path(args.summary).read_text())
    assert result["status"] == "independently_audited"
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    conditions = [
        ("mcar30", "reference"),
        ("mcar30", "observed"),
        ("mar", "reference"),
        ("mar", "observed"),
    ]
    labels = ["Complete\n(MCAR scale)", "MCAR 30%", "Complete\n(MAR scale)", "MAR"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    plotted = []
    for row, n in enumerate((64, 256)):
        for position, (mask, arm) in enumerate(conditions):
            cell = next(
                c
                for c in result["cells"]
                if c["origin"] == "fresh"
                and c["budget"] == "original"
                and c["case"] == "four_groups"
                and c["n"] == n
                and c["mask"] == mask
                and c["arm"] == arm
            )
            rate = cell["rates"]["correct_k"]
            total = rate["successful"]
            fraction = rate["count"] / total if total else 0
            ci = rate["conditional_exact95"]
            axes[row, 0].bar(position, fraction, width=0.62, color="#0072B2")
            if ci:
                axes[row, 0].errorbar(
                    position,
                    fraction,
                    yerr=[[fraction - ci[0]], [ci[1] - fraction]],
                    color="#333333",
                    capsize=3,
                    fmt="none",
                    linewidth=1,
                )
            axes[row, 0].text(
                position,
                0.05,
                f"{rate['count']}/{total}"
                + (f"\n{cell['failures']} withheld" if cell["failures"] else ""),
                ha="center",
                color="white" if fraction > 0.12 else "#333333",
                fontsize=9,
            )
            axes[row, 1].plot(
                position - 0.08,
                cell["median_selected_ari"],
                "o",
                color="#0072B2",
                label="Selected partition" if position == 0 else None,
            )
            if cell["median_eligible_k4_ari"] is not None:
                axes[row, 1].plot(
                    position + 0.08,
                    cell["median_eligible_k4_ari"],
                    "D",
                    color="#D55E00",
                    label="Eligible K=4 candidate" if position == 0 else None,
                )
            plotted.append(
                dict(
                    n=n,
                    mask=mask,
                    arm=arm,
                    correct_k=rate,
                    median_selected_ari=cell["median_selected_ari"],
                    median_eligible_k4_ari=cell["median_eligible_k4_ari"],
                    failures=cell["failures"],
                )
            )
        for col in range(2):
            ax = axes[row, col]
            ax.set_xticks(np.arange(4), labels, fontsize=9)
            ax.set_ylim(0, 1.08)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
        axes[row, 0].set_title(f"N={n}: BIC selects four components")
        axes[row, 0].set_ylabel("Fraction of resolved searches")
        axes[row, 1].set_title(f"N={n}: generating-label agreement")
        axes[row, 1].set_ylabel("Median adjusted Rand index")
        axes[row, 1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Observed-data GMM: candidate recovery and BIC selection", fontsize=14)
    fig.supxlabel(
        "Ten independent datasets per N cell; masks/references are paired. "
        "Bars: pointwise exact 95% intervals.",
        fontsize=9,
    )
    for extension in ("png", "pdf", "svg"):
        target = root / f"recovery-selection.{extension}"
        fig.savefig(target, dpi=180)
        if extension == "svg":
            target.write_text(
                "\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n"
            )
    plt.close(fig)
    (root / "figure-data.json").write_text(json.dumps(plotted, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
