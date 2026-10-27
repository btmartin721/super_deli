"""Reproduce scientific figures from small audited aggregate findings."""

import argparse
from pathlib import Path


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from super_deli.io.config import read_json

    p = argparse.ArgumentParser()
    p.add_argument("summary", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    rows = read_json(a.summary)["rates"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True, layout="constrained")
    order = [
        ("gmm", "complete"),
        ("gmm", "complete_case"),
        ("gmm", "mean"),
        ("density", "complete"),
        ("density", "complete_case"),
        ("density", "mean"),
    ]
    labels = [
        "GMM\nreference",
        "GMM\ncomplete",
        "GMM\nmean",
        "Density\nreference",
        "Density\ncomplete",
        "Density\nmean",
    ]
    colors = ["#777777", "#0072B2", "#E69F00", "#777777", "#0072B2", "#E69F00"]
    for ri, case in enumerate(("gaussian", "four_spherical")):
        for ci, n in enumerate((64, 256)):
            ax = axes[ri, ci]
            for j, (arm, policy) in enumerate(order):
                mask = "oracle" if policy == "complete" else "mcar30"
                r = next(
                    r
                    for r in rows
                    if (r["case"], r["n"], r["mask"], r["policy"], r["arm"])
                    == (case, n, mask, policy, arm)
                )
                rate = r["conditional_rate"]
                interval = r["pointwise_exact_95"]
                if rate is not None:
                    ax.errorbar(
                        j,
                        100 * rate,
                        yerr=[[100 * (rate - interval[0])], [100 * (interval[1] - rate)]],
                        fmt="o",
                        color=colors[j],
                        capsize=4,
                    )
                else:
                    ax.text(j, 5, "unavailable", rotation=90, ha="center", fontsize=7)
                if r["failed_or_blocked"]:
                    ax.text(j, 106, f"{r['successful']}/10", ha="center", fontsize=7)
            ax.set(
                xticks=range(6),
                xticklabels=labels,
                ylim=(-3, 113),
                yticks=[0, 25, 50, 75, 100],
                title=("Gaussian: K > 1" if case == "gaussian" else "Four groups: K = 4")
                + f"; N={n}",
                ylabel="Datasets (%)",
            )
            ax.grid(axis="y", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="x", labelsize=7)
    fig.suptitle("30% independent missingness: selection and uncertainty", fontsize=13)
    fig.supxlabel(
        "Complete-data reference versus fixed incomplete-data policies; "
        "n=10 independent datasets/cell.\n"
        "Intervals are pointwise and conditional on successful fits; "
        "fractions mark incomplete cells.",
        fontsize=9,
    )
    for ext in ("png", "pdf", "svg"):
        fig.savefig(a.output_dir / ("missingness-comparison." + ext), dpi=180)
    svg = a.output_dir / "missingness-comparison.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
