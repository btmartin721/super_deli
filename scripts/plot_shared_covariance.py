"""Plot the audited paired recovery gains and prespecified stress limitations."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot(audit, destination):
    result = json.loads((audit / "summary.json").read_text())
    lookup = {(r["cohort"], r["family"]): r for r in result["summaries"]}
    cohorts = ("archived", "fresh_64", "fresh_256", "unequal_variance")
    labels = ("Archived\nN=64", "Fresh\nN=64", "Fresh\nN=256", "Unequal variance\nN=64")
    colors = {"full": "#487AA1", "tied": "#009E73"}
    plt.rcParams.update(
        {
            "font.size": 9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(13, 4.6), layout="constrained", gridspec_kw={"width_ratios": [1.2, 1.2, 1]}
    )
    x = np.arange(4)
    for family, offset in (("full", -0.18), ("tied", 0.18)):
        counts = np.array([lookup[c, family]["selected_k_counts"]["4"] for c in cohorts])
        intervals = np.array([lookup[c, family]["selected_four_wilson_95"] for c in cohorts]).T * 25
        bars = axes[0].bar(
            x + offset,
            counts,
            width=0.34,
            color=colors[family],
            label="Separate full" if family == "full" else "Shared covariance",
        )
        axes[0].errorbar(
            x + offset,
            counts,
            yerr=np.maximum(0, [counts - intervals[0], intervals[1] - counts]),
            fmt="none",
            ecolor="#333333",
            capsize=3,
            lw=0.8,
        )
        for bar, count in zip(bars, counts, strict=True):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2, count + 0.5, str(count), ha="center", fontsize=8
            )
        means = [lookup[c, family]["mean_ari"] for c in cohorts]
        axes[1].bar(x + offset, means, width=0.34, color=colors[family])
    axes[0].set(
        title="A  Selected K=4", ylabel="Datasets selecting four components / 25", ylim=(0, 29)
    )
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    axes[1].set(
        title="B  Agreement with generating labels",
        ylabel="Mean adjusted Rand agreement",
        ylim=(0, 1),
    )
    for ax in axes[:2]:
        ax.set_xticks(x, labels)
    categories = [
        ("zero_association", "full"),
        ("zero_association", "tied"),
        ("cline", "full"),
        ("cline", "tied"),
    ]
    left = np.zeros(4)
    for k, color in zip(range(1, 5), ("#E6E6E6", "#56B4E9", "#E69F00", "#CC79A7"), strict=True):
        values = np.array([lookup[c, f]["selected_k_counts"][str(k)] for c, f in categories])
        present = values > 0
        axes[2].barh(
            np.arange(4)[present], values[present], left=left[present], color=color, label=f"K={k}"
        )
        for i, (value, start) in enumerate(zip(values, left, strict=True)):
            if value:
                axes[2].text(start + value / 2, i, str(value), ha="center", va="center", fontsize=8)
        left += values
    axes[2].set_yticks(
        range(4),
        (
            "Zero association: full",
            "Zero association: shared",
            "Continuous cline: full",
            "Continuous cline: shared",
        ),
    )
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, 25)
    axes[2].set_title("C  Selection without discrete groups", fontsize=10)
    axes[2].set_xlabel("Datasets / 25")
    axes[2].legend(
        frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=8
    )
    for ax in axes:
        ax.title.set_fontsize(10)
        ax.title.set_weight("bold")
    fig.suptitle(
        "Shared covariance improves four-group recovery, with stress limitations",
        fontsize=13,
        weight="bold",
    )
    fig.supxlabel(
        "Frozen two-PC settings; pointwise Wilson 95% intervals in A. "
        "Shared covariance remains opt-in.",
        fontsize=9,
    )
    destination.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        path = destination / f"shared-covariance.{ext}"
        fig.savefig(path, dpi=150, metadata={"Creator": "Super Deli paired covariance study"})
        if ext == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot(args.audit, args.output_dir)
