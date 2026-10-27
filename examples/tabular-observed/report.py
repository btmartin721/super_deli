"""Plot only audited small aggregates; no fitting or model selection."""

import argparse
import json
from pathlib import Path


def plot(source, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    aggregate = json.loads(Path(source).read_text())
    cells = aggregate["cells"]
    assert aggregate["status"] == "audited" and len(cells) == 120
    arms = ("observed", "complete_case", "mean", "reference")
    colors = ("#0072B2", "#E69F00", "#D55E00", "#009E73")
    labels = ("Observed-data EM", "Complete cases", "Mean-filled", "Complete-data reference")
    cases = ("spherical", "correlated", "four_groups")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True, constrained_layout=True)
    figure_data = []
    for i, n in enumerate((64, 256)):
        for j, mask in enumerate(("mcar30", "mar")):
            ax = axes[i, j]
            for k, arm in enumerate(arms):
                rows = [
                    next(
                        r
                        for r in cells
                        if r["case"] == case
                        and r["n"] == n
                        and r["mask"] == mask
                        and r["arm"] == arm
                    )
                    for case in cases
                ]
                value = np.array([r["rates"]["correct_k"]["count"] for r in rows])
                missing = np.array([r["failed"] for r in rows])
                location = np.arange(3) + (k - 1.5) * 0.19
                ax.bar(location, value, width=0.18, color=colors[k], label=labels[k])
                ax.bar(
                    location,
                    missing,
                    bottom=value,
                    width=0.18,
                    facecolor="none",
                    edgecolor=colors[k],
                    hatch="///",
                )
                figure_data.extend(
                    dict(n=n, mask=mask, case=case, arm=arm, count=int(v), failed=int(f))
                    for case, v, f in zip(cases, value, missing, strict=True)
                )
            ax.set_xticks(
                range(3),
                [
                    "Spherical\n(target K=1)",
                    "Correlated\n(target K=1)",
                    "Four groups\n(target K=4)",
                ],
            )
            ax.set_title(
                f"N={n}; {'30% random missingness' if mask == 'mcar30' else 'observed-trait MAR'}"
            )
            ax.set_ylim(0, 10.7)
            ax.set_yticks(range(0, 11, 2))
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel("Selections matching generating K / 10 datasets")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside upper center", ncol=4, frameon=False)
    fig.supxlabel(
        "Hatching: failed/unresolved searches. Each cell uses 10 independent datasets; "
        "masks and arms are paired.",
        fontsize=9,
    )
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        output = destination / f"missing-data-comparison.{ext}"
        fig.savefig(output, dpi=180)
        if ext == "svg":
            output.write_text(
                "\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n"
            )
    plt.close(fig)
    return figure_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("summary")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.summary, args.output_dir), indent=2))
