"""Export audited cell aggregates and an all-settings figure; never individual data."""

import argparse
import shutil
from pathlib import Path

from super_deli.io.config import read_json, write_json


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from super_deli.benchmarking.view_signal_config import REPRESENTATIONS
    from super_deli.benchmarking.view_signal_study import check_frozen

    parser = argparse.ArgumentParser()
    parser.add_argument("study_root", type=Path)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root, repo = args.study_root.resolve(), args.repository.resolve()
    frozen, _ = check_frozen(root)
    audit = read_json(root / "main-audit.json")
    if not audit["passed"] or audit["failed"] or audit["unavailable"]:
        raise ValueError("Complete independently audited diagnostic required")
    output = repo / "examples/matched-view-signal/results"
    output.mkdir(exist_ok=False)
    for name in ("summary", "all-spaces", "decoder-summary", "archived-gmm-summary"):
        shutil.copyfile(root / (name + ".json"), output / (name + ".json"))
    write_json(
        {
            "preflight": read_json(root / "preflight-audit.json"),
            "main": audit,
            "freeze_sha256": audit["freeze_sha256"],
            "source_file_count": len(frozen["source_sha256"]),
            "source_parent": "547a74035e4f7ebf5fe371f16e16d626fa1a95ae",
        },
        output / "independent-audit.json",
    )
    rows = read_json(root / "all-spaces.json")
    cases = ("morphology_only", "four_complementary", "four_redundant")
    cells = [(case, n) for case in cases for n in (64, 256)]
    matrix = np.array(
        [
            [
                next(
                    r["k4_ari_ge_0_8"]
                    for r in rows
                    if r["representation"] == name and (r["case"], r["n"]) == cell
                )
                for cell in cells
            ]
            for name in REPRESENTATIONS
        ]
    )
    fig, ax = plt.subplots(figsize=(10, 10.5), layout="constrained")
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=10, aspect="auto")
    ax.set_yticks(
        range(len(REPRESENTATIONS)), [n.replace("_", " ") for n in REPRESENTATIONS], fontsize=9
    )
    labels = {
        "morphology_only": "Only morphology\nhas signal",
        "four_complementary": "Complementary\nsignal",
        "four_redundant": "Redundant\nsignal",
    }
    ax.set_xticks(range(6), [labels[c] + f"\nN={n}" for c, n in cells], fontsize=9)
    ax.set_xlabel("Archived generating law and matched-individual sample size", labelpad=12)
    ax.set_ylabel("Prespecified coordinate space and neural seed")
    ax.set_title(
        "Uninformative ecology suppresses recovery before and after compression\n"
        "K=4 and full ARI ≥ 0.8; count out of 10 reused datasets",
        fontsize=12,
        pad=14,
    )
    for i in range(len(REPRESENTATIONS)):
        for j in range(6):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] < 6 else "black",
                fontsize=10,
            )
    for line in (8.5, 14.5):
        ax.axhline(line, color="white", linewidth=1.1)
    fig.colorbar(
        image, ax=ax, fraction=0.035, label="Successful recoveries / 10", ticks=range(0, 11, 2)
    )
    figures = repo / "docs/figures/matched-view-signal"
    figures.mkdir(parents=True, exist_ok=False)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(figures / ("all-spaces-recovery." + extension), dpi=160)
    plt.close(fig)
    write_json(
        {
            "representations": list(REPRESENTATIONS),
            "cells": [list(c) for c in cells],
            "counts": matrix.tolist(),
            "denominator": 10,
            "interpretation": "Exploratory reused observations; seeds/spaces are repeated measures",
        },
        figures / "figure-data.json",
    )


if __name__ == "__main__":
    main()
