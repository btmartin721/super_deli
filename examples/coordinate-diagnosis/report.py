"""Export audited cell aggregates and fixed all-seed figures; detailed evidence stays local."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True, type=Path)
    root = parser.parse_args().study.resolve()
    repo = Path(__file__).resolve().parents[2]

    def read(name):
        return json.loads((root / (name + ".json")).read_text())

    assert read("main-audit")["passed"] and not (root / "STOP.json").exists()
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=False)
    for name in (
        "summary",
        "all-groups",
        "primary-rates",
        "paired-comparisons",
        "hierarchy-groups",
        "candidate-groups",
    ):
        value = read(name)
        assert not (isinstance(value, dict) and "records" in value)
        (out / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")
    (out / "independent-audit.json").write_text(
        json.dumps(read("summary")["audits"], indent=2) + "\n"
    )
    groups = read("all-groups")
    cases = (
        "morphology_only",
        "ecology_only",
        "four_complementary",
        "four_redundant",
        "gaussian_independent",
        "gaussian_correlated",
    )
    cells = [(case, n) for case in cases for n in (64, 256)]
    arms = [
        f"separate_{seed}__{method}"
        for seed in (0, 101, 202)
        for method in ("hdbscan", "gmm_full", "gmm_shared")
    ]
    counts = np.array(
        [
            [
                next(
                    x["endpoint_count"]
                    for x in groups
                    if (x["case"], x["n"], x["arm"]) == (case, n, arm)
                )
                for case, n in cells
            ]
            for arm in arms
        ]
    )
    fig, ax = plt.subplots(figsize=(13.2, 6.2))
    mask = np.tile(np.arange(12) >= 8, (9, 1))
    ax.imshow(np.ma.masked_where(mask, counts), vmin=0, vmax=5, cmap="Blues", aspect="auto")
    ax.imshow(np.ma.masked_where(~mask, counts), vmin=0, vmax=5, cmap="Oranges", aspect="auto")
    for i, j in np.ndindex(counts.shape):
        ax.text(
            j,
            i,
            str(counts[i, j]),
            ha="center",
            va="center",
            color="white" if counts[i, j] >= 4 else "black",
        )
    names = (
        "Morphology\nonly",
        "Ecology\nonly",
        "Complementary",
        "Redundant",
        "Gaussian\nindependent",
        "Gaussian\ncorrelated",
    )
    ax.set_xticks(
        range(12), [f"{names[i // 2]}\nN={n}" for i, (_, n) in enumerate(cells)], fontsize=7
    )
    methods = {"hdbscan": "HDBSCAN", "gmm_full": "Full GMM", "gmm_shared": "Shared GMM"}
    ax.set_yticks(
        range(9),
        [f"Seed {seed} | {methods[method]}" for seed in (0, 101, 202) for method in methods],
        fontsize=9,
    )
    for x in (1.5, 3.5, 5.5, 7.5, 9.5):
        ax.axvline(x, color="white", linewidth=2)
    ax.axvline(7.5, color="#a63b2e", linewidth=3)
    ax.set_title("Separate neural coordinates held fixed: clustering-stage diagnosis", pad=14)
    fig.text(
        0.5,
        0.035,
        "Blue: K=4 and full ARI ≥0.8. Orange: K>1 Gaussian splits. "
        "Counts /5 archived datasets.\nEvery seed retained; retrospective comparison, "
        "no automatic model/default selection.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.20, right=0.99, top=0.90, bottom=0.24)
    figures = repo / "docs/figures/coordinate-diagnosis"
    figures.mkdir(parents=True, exist_ok=False)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(figures / ("frozen-coordinate-recovery." + suffix), dpi=180)
    plt.close(fig)
    svg = figures / "frozen-coordinate-recovery.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    (figures / "figure-data.json").write_text(
        json.dumps(dict(arms=arms, cells=cells, counts=counts.tolist()), indent=2) + "\n"
    )
    print(
        json.dumps(
            dict(
                passed=True,
                public_group_rows=len(groups),
                primary_rates=len(read("primary-rates")),
                paired_contrasts=len(read("paired-comparisons")),
                figure_cells=counts.size,
            )
        )
    )


if __name__ == "__main__":
    main()
