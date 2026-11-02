"""Export only small aggregates and static figures from an independently audited study."""

import argparse
from pathlib import Path


def export(root, repository):
    import matplotlib.pyplot as plt
    import numpy as np

    from super_deli.io.config import read_json, write_json

    result = read_json(root / "aggregate.json")
    audit = read_json(root / "main-audit.json")
    assert audit["passed"] and not result["fixture"]
    public = repository / "examples/gaussian-reference-spectral/results"
    figures = repository / "docs/figures/gaussian-reference-spectral"
    public.mkdir(parents=True, exist_ok=False)
    figures.mkdir(parents=True, exist_ok=False)
    for name, key in [
        ("all-groups", "groups"),
        ("primary-rates", "primary_rates"),
        ("paired-comparisons", "paired_comparisons"),
    ]:
        write_json(result[key], public / (name + ".json"))
    write_json(
        {
            k: v
            for k, v in result.items()
            if k not in {"groups", "primary_rates", "paired_comparisons"}
        },
        public / "summary.json",
    )
    write_json(
        {
            "preflight": read_json(root / "preflight-audit.json"),
            "main": audit,
            "post_fit_arithmetic": read_json(root / "report-qa.json"),
        },
        public / "independent-audit.json",
    )
    arms = ["spectral_5", "reference_5", "spectral_15", "reference_15", "gmm", "density"]
    labels = [
        "Largest gap: 5",
        "Gaussian reference: 5",
        "Largest gap: 15",
        "Gaussian reference: 15",
        "PCA/GMM",
        "PCA/HDBSCAN",
    ]
    colors = ["#999999", "#0072B2", "#BBBBBB", "#D55E00", "#009E73", "#CC79A7"]
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "super-deli-gaussian-reference-v1",
        }
    )
    data = []
    for filename, cases, endpoint, title in [
        (
            "gaussian-controls",
            ["gaussian", "gaussian_8d"],
            "nontrivial_return",
            "Returned multiple groups from one Gaussian cloud",
        ),
        (
            "group-recovery",
            ["moons", "four_spherical", "four_unequal"],
            "returned_correct_k_ari08",
            "Correct group count and strong membership agreement (ARI >= 0.8)",
        ),
    ]:
        fig, axes = plt.subplots(len(cases), 2, figsize=(10, 3.1 * len(cases)), sharex=True)
        for row, case in enumerate(cases):
            for column, n in enumerate((64, 256)):
                ax = axes[row, column]
                for condition, offset, filled in [("clean", -0.17, True), ("noisy", 0.17, False)]:
                    for j, arm in enumerate(arms):
                        item = next(
                            r
                            for r in result["primary_rates"]
                            if (r["case"], r["n"], r["condition"], r["arm"], r["endpoint"])
                            == (case, n, condition, arm, endpoint)
                        )
                        data.append(item)
                        value = item["rate"]
                        lo, hi = item["marginal_95_interval"]
                        y = j + offset
                        ax.errorbar(
                            value,
                            y,
                            xerr=[[value - lo], [hi - value]],
                            color=colors[j],
                            marker="o",
                            markersize=5,
                            markerfacecolor=colors[j] if filled else "white",
                            capsize=2,
                            linewidth=1,
                        )
                        ax.text(
                            1.025,
                            y,
                            f"{item['count']}/{item['requested']}"
                            + (
                                f"  ({item['unavailable']} withheld)" if item["unavailable"] else ""
                            ),
                            fontsize=7,
                            va="center",
                        )
                ax.set_title(f"{case.replace('_', ' ')}; N={n}")
                ax.set_yticks(np.arange(6), labels if column == 0 else [""] * 6)
                ax.set_ylim(5.6, -0.6)
                ax.set_xlim(-0.025, 1.52)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
                ax.axvline(0, color="#DDDDDD", linewidth=0.6)
                ax.axvline(1, color="#DDDDDD", linewidth=0.6)
                ax.grid(axis="x", alpha=0.15)
                ax.set_xlabel("Fraction of all requested datasets")
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, 0.05, 1, 0.96))
        fig.text(
            0.02,
            0.018,
            "Filled: clean; open: noisy. Ten independent datasets/cell; paired conditions. "
            "Bars: marginal 95% exact intervals.\nSimultaneous 192-rate intervals in exports. "
            "Withheld results are not successful returned partitions. No species guarantee.",
            fontsize=8,
        )
        for extension in ("png", "pdf", "svg"):
            fig.savefig(
                figures / f"{filename}.{extension}",
                dpi=180,
                metadata={"Creator": "Super Deli"} if extension == "pdf" else None,
            )
        plt.close(fig)
    write_json(data, figures / "figure-data.json")
    for path in figures.glob("*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    print("Exported aggregate-only tables and figures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Completed local study root")
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()
    export(args.output_dir, args.repository)
