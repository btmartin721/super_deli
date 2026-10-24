"""Fixed power/recovery figure and small aggregate table from audited study output."""

import argparse
import csv
import json
from pathlib import Path


def report(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = summary["cells"]
    groups = [r for r in rows if r["has_groups"]]
    rates = {(r["cell"], r["endpoint"]): r for r in summary["primary_rates"]}
    table = []
    for row in rows:
        record = {k: row[k] for k in ("cell", "law", "signal", "n_units", "requested")}
        for criterion in ("aic", "bic"):
            record[criterion + "_selected_k_counts"] = json.dumps(
                row[criterion + "_selected_k_counts"], sort_keys=True
            )
            record[criterion + "_ari_mean"] = row[criterion + "_ari"]["mean"]
            record[criterion + "_ari_mcse"] = row[criterion + "_ari"]["mcse_mean"]
        record["likelihood_rejected"] = row["likelihood_rejection"]["events"]
        record["likelihood_available"] = row["likelihood_rejection"]["available"]
        record["bic_rejected"] = row["bic_rejection"]["events"]
        record["bic_available"] = row["bic_rejection"]["available"]
        table.append(record)
    with (output / "aggregate.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), layout="constrained")
    endpoints = [
        ("likelihood_rejection", "Gaussian-departure detection"),
        ("aic_k4", "AIC selects four components"),
        ("joint_rejection_and_aic_k4", "Both rejection and AIC K=4"),
    ]
    colors = ["#17678c", "#b45118", "#7164a4", "#2d7848"]
    sample_sizes = sorted({r["n_units"] for r in groups})
    for j, n in enumerate(sample_sizes):
        cohort = sorted((r for r in groups if r["n_units"] == n), key=lambda r: r["signal"])
        x = np.asarray([r["signal"] for r in cohort])
        color = colors[j % len(colors)]
        for ax, (endpoint, title) in zip(axes.flat[:3], endpoints, strict=True):
            selected = [rates[r["cell"], endpoint] for r in cohort]
            y = (
                np.asarray(
                    [
                        r["rate_among_available"]
                        if r["rate_among_available"] is not None
                        else np.nan
                        for r in selected
                    ]
                )
                * 100
            )
            for field, width, alpha in (
                ("bonferroni_familywise_95", 5, 0.18),
                ("exact_interval_95", 1.2, 1),
            ):
                lo = np.asarray([r[field][0] if r[field] else np.nan for r in selected]) * 100
                hi = np.asarray([r[field][1] if r[field] else np.nan for r in selected]) * 100
                ax.errorbar(
                    x,
                    y,
                    yerr=np.vstack((y - lo, hi - y)),
                    fmt="none",
                    color=color,
                    elinewidth=width,
                    alpha=alpha,
                    capsize=0,
                )
            ax.plot(x, y, "o-", color=color, label=f"N={n}")
            ax.set_title(title)
            ax.set_ylim(-3, 103)
            ax.set_ylabel("Observed datasets (%)")
        for criterion, style in (("aic", "-"), ("bic", "--")):
            y = [r[criterion + "_ari"]["mean"] for r in cohort]
            error = [r[criterion + "_ari"]["mcse_mean"] or 0 for r in cohort]
            axes[1, 1].errorbar(
                x,
                y,
                yerr=error,
                color=color,
                linestyle=style,
                marker="o" if criterion == "aic" else "s",
                label=f"N={n}, {criterion.upper()}",
            )
    axes[1, 1].set_title("Unconditional assignment agreement")
    axes[1, 1].set_ylabel("Mean adjusted Rand agreement ± MCSE")
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[0, 0].legend(frameon=False)
    for ax in axes.flat:
        ax.set_xlabel("Raw signal multiplier")
        ax.grid(axis="y", alpha=0.2)
    figure.suptitle("Fixed grouped AIC/likelihood validation: generating partitions, not species")
    figure.supxlabel(
        "Rate bars: thin pointwise exact 95%; pale simultaneous exact 95% across 18 primary rates",
        fontsize=9,
    )
    for extension in ("png", "pdf", "svg"):
        figure.savefig(output / f"grouped-power-recovery.{extension}", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report(json.loads(args.summary.read_text()), args.output_dir)


if __name__ == "__main__":
    main()
