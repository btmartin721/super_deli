"""Render the audited fixed-grid recovery outcomes and baseline BIC tradeoff."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

VARIANTS = ("baseline", "pcs_3", "pcs_4", "ecology_0.5", "ecology_2", "ecology_4", "starts_100")
LABELS = (
    "Baseline",
    "3 joint PCs",
    "4 joint PCs",
    "Ecology ×0.5",
    "Ecology ×2",
    "Ecology ×4",
    "100 starts",
)
COHORTS = ("archived", "fresh_64", "fresh_256")
NAMES = ("Archived\nN=64", "Fresh\nN=64", "Fresh\nN=256")
COLORS = ("#0072B2", "#D55E00", "#009E73")


def plot(root, destination):
    summary = json.loads((root / "summary.json").read_text())
    lookup = {(r["cohort"], r["variant"]): r for r in summary["summaries"]}
    counts = np.array([[lookup[c, v]["selected_k_counts"]["4"] for c in COHORTS] for v in VARIANTS])
    agreement = np.array([[lookup[c, v]["ari"]["median"] for c in COHORTS] for v in VARIANTS])
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
        1, 3, figsize=(12.4, 4.5), gridspec_kw={"width_ratios": [1.1, 1, 1.7]}, layout="constrained"
    )
    for ax, values, title, limits in zip(
        axes[:2],
        (counts, agreement),
        ("A  Selected K=4", "B  Median adjusted Rand agreement"),
        ((0, 25), (0, 1)),
        strict=True,
    ):
        ax.imshow(values, cmap="Blues", vmin=limits[0], vmax=limits[1], aspect="auto")
        for i in range(7):
            for j in range(3):
                label = f"{values[i, j]:.0f}/25" if ax is axes[0] else f"{values[i, j]:.2f}"
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    color="white" if values[i, j] > 0.58 * limits[1] else "#17212B",
                )
        ax.set_xticks(range(3), NAMES)
        ax.set_yticks(range(7), LABELS if ax is axes[0] else [""] * 7)
        ax.tick_params(length=0)
        ax.set_title(title, loc="left", pad=12, fontsize=10, weight="bold")
    with (root / "candidates.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = {}
    for row in rows:
        if row["variant"] == "baseline":
            values[(row["cohort"], row["repetition"], int(row["k"]))] = row
    ax = axes[2]
    maximum = 0
    for cohort, color in zip(COHORTS, COLORS, strict=True):
        for eligible, marker in ((True, "o"), (False, "x")):
            x, y = [], []
            for i in range(25):
                k2, k4 = values[cohort, str(i), 2], values[cohort, str(i), 4]
                if (k4["status"] == "eligible") != eligible:
                    continue
                x.append(float(k4["penalty"]) - float(k2["penalty"]))
                y.append(
                    2
                    * (float(k4["log_likelihood_from_bic"]) - float(k2["log_likelihood_from_bic"]))
                )
            maximum = max(maximum, max(y, default=0), max(x, default=0))
            ax.scatter(
                x,
                y,
                c=color,
                marker=marker,
                s=25,
                alpha=0.72,
                label=cohort.replace("_", " ") if eligible else None,
            )
    ax.plot([0, maximum * 1.04], [0, maximum * 1.04], c="#555555", ls="--", lw=1)
    ax.set_xlim(0, maximum * 1.06)
    ax.set_ylim(0, maximum * 1.06)
    ax.set_xlabel("Extra BIC penalty: K=4 versus K=2")
    ax.set_ylabel("Twice log-likelihood gain: K=4 versus K=2")
    ax.set_title(
        "C  Baseline likelihood gain and BIC cost", loc="left", pad=12, fontsize=10, weight="bold"
    )
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.text(
        0.02,
        0.98,
        "Above line: K=4 beats K=2 BIC\n×: K=4 fails minimum size",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    fig.suptitle("Four-group recovery under the frozen diagnostic grid", fontsize=13, weight="bold")
    fig.supxlabel(
        "25 datasets per cohort; configurations remain diagnostic. "
        "No bootstrap or significance reruns.",
        fontsize=9,
    )
    destination.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        path = destination / f"integrative-recovery.{extension}"
        fig.savefig(path, dpi=150, metadata={"Creator": "Super Deli recovery diagnostics"})
        if extension == "svg":
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
