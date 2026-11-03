"""Export audited cell aggregates and figures, never raw records or trained weights."""

import argparse
from pathlib import Path


def export(root, repository):
    import matplotlib.pyplot as plt
    import numpy as np

    from super_deli.benchmarking.matched_config import ARMS
    from super_deli.io.config import read_json, write_json

    result = read_json(root / "aggregate.json")
    assert not result["fixture"] and read_json(root / "report-qa.json")["passed"]
    public = repository / "examples/matched-neural/results"
    figures = repository / "docs/figures/matched-neural"
    public.mkdir(parents=True, exist_ok=False)
    figures.mkdir(parents=True, exist_ok=False)
    excluded = {"groups", "primary_rates", "paired_comparisons", "stability"}
    for name, key in (
        ("all-groups", "groups"),
        ("primary-rates", "primary_rates"),
        ("paired-comparisons", "paired_comparisons"),
        ("native-stability", "stability"),
    ):
        write_json(result[key], public / f"{name}.json")
    write_json({k: v for k, v in result.items() if k not in excluded}, public / "summary.json")
    write_json(
        {
            "preflight": read_json(root / "preflight-audit.json"),
            "main": read_json(root / "main-audit.json"),
            "post_fit_arithmetic": read_json(root / "report-qa.json"),
        },
        public / "independent-audit.json",
    )
    labels = [
        f"{view} AE: seed {seed}"
        for view in ("Paired", "Morphology", "Ecology")
        for seed in (0, 101, 202)
    ] + [
        "Integrative PCA/GMM",
        "Morphology PCA/GMM",
        "Ecology PCA/GMM",
        "Morphology PCA/HDBSCAN",
        "Ecology PCA/HDBSCAN",
    ]
    colors = ["#0072B2"] * 3 + ["#009E73"] * 3 + ["#D55E00"] * 3 + ["#555555"] * 5
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "matched-neural-v1",
        }
    )
    data = []
    for filename, cases, endpoint, title in (
        (
            "gaussian-controls",
            ("gaussian_independent", "gaussian_correlated"),
            "nontrivial_return",
            "Multiple groups returned from a single Gaussian population",
        ),
        (
            "four-group-recovery",
            ("four_redundant", "four_complementary", "morphology_only"),
            "returned_correct_k_ari08",
            "Four groups returned with strong membership agreement (ARI ≥ 0.8)",
        ),
    ):
        fig, axes = plt.subplots(len(cases), 2, figsize=(12, 4.4 * len(cases)), sharex=True)
        for row, case in enumerate(cases):
            for column, n in enumerate((64, 256)):
                ax = axes[row, column]
                for j, arm in enumerate(ARMS):
                    item = next(
                        r
                        for r in result["primary_rates"]
                        if (r["case"], r["n"], r["arm"], r["endpoint"]) == (case, n, arm, endpoint)
                    )
                    data.append(item)
                    value = item["rate"]
                    lo, hi = item["marginal_95_interval"]
                    ax.errorbar(
                        value,
                        j,
                        xerr=[[value - lo], [hi - value]],
                        color=colors[j],
                        marker="o",
                        markersize=4,
                        capsize=2,
                    )
                    ax.text(1.035, j, f"{item['count']}/{item['requested']}", va="center")
                ax.set_title(f"{case.replace('_', ' ')}; N={n}")
                ax.set_yticks(np.arange(len(ARMS)), labels if column == 0 else [""] * len(ARMS))
                ax.set_ylim(len(ARMS) - 0.5, -0.6)
                ax.set_xlim(-0.025, 1.18)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
                ax.grid(axis="x", alpha=0.15)
                ax.set_xlabel("Fraction of all requested datasets")
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, 0.055, 1, 0.965))
        fig.text(
            0.02,
            0.018,
            "Ten independent datasets/cell; neural seeds are paired repeated measurements. "
            "Bars: marginal 95% exact intervals.\n"
            "All 224 simultaneous intervals and unavailable counts are in the tables. "
            "No winning seed, new default or species guarantee.",
            fontsize=8,
        )
        for extension in ("png", "pdf", "svg"):
            fig.savefig(
                figures / f"{filename}.{extension}",
                dpi=160,
                metadata={"Creator": "Super Deli"} if extension == "pdf" else None,
            )
        plt.close(fig)
    write_json(data, figures / "figure-data.json")
    for path in figures.glob("*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    print("Exported aggregate-only tables and figures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()
    export(args.output_dir, args.repository)
