"""Export audited small aggregates and a fixed all-seed architecture figure."""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True, type=Path)
    args = parser.parse_args()
    root = args.study.resolve()
    repo = Path(__file__).resolve().parents[2]

    def read(path):
        return json.loads(path.read_text())

    assert read(root / "main-audit.json")["passed"]
    target = Path(__file__).parent / "results"
    target.mkdir(exist_ok=False)
    for name in (
        "all-groups",
        "primary-rates",
        "paired-comparisons",
        "native-stability",
    ):
        shutil.copy2(root / (name + ".json"), target / (name + ".json"))

    def aggregate_audit(value):
        return {key: item for key, item in value.items() if key != "records"}

    summary = read(root / "summary.json")
    summary["main_audit"] = aggregate_audit(summary["main_audit"])
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (target / "independent-audit.json").write_text(
        json.dumps(
            {
                phase: aggregate_audit(read(root / (phase + "-audit.json")))
                for phase in ("preflight", "main")
            },
            indent=2,
        )
        + "\n"
    )
    groups = read(root / "all-groups.json")
    cases = [
        "morphology_only",
        "ecology_only",
        "four_complementary",
        "four_redundant",
        "gaussian_independent",
        "gaussian_correlated",
    ]
    arms = [
        f"{weight}_{seed}" for weight in ("separate", "joint2", "joint4") for seed in (0, 101, 202)
    ]
    cells = [(case, n) for case in cases for n in (64, 256)]
    values = np.array(
        [
            [
                next(
                    r["endpoint_count"]
                    for r in groups
                    if (r["case"], r["n"], r["arm"]) == (case, n, arm)
                )
                for case, n in cells
            ]
            for arm in arms
        ]
    )
    fig, ax = plt.subplots(figsize=(12.8, 6.0))
    recovery = np.ma.masked_where(np.tile(np.arange(12) >= 8, (9, 1)), values)
    splitting = np.ma.masked_where(np.tile(np.arange(12) < 8, (9, 1)), values)
    ax.imshow(recovery, vmin=0, vmax=5, cmap="Blues", aspect="auto")
    ax.imshow(splitting, vmin=0, vmax=5, cmap="Oranges", aspect="auto")
    for i, j in np.ndindex(values.shape):
        ax.text(
            j,
            i,
            str(values[i, j]),
            ha="center",
            va="center",
            color="white" if values[i, j] >= 4 else "black",
        )
    names = [
        "Morphology\nonly",
        "Ecology\nonly",
        "Complementary",
        "Redundant",
        "Gaussian\nindependent",
        "Gaussian\ncorrelated",
    ]
    ax.set_xticks(
        range(12), [f"{names[i // 2]}\nN={n}" for i, (_, n) in enumerate(cells)], fontsize=7
    )
    ax.set_yticks(
        range(9),
        [
            f"{fraction} | seed {seed}"
            for fraction in ("Separate 2+2", "Joint 2", "Joint 4")
            for seed in (0, 101, 202)
        ],
        fontsize=9,
    )
    for x in (1.5, 3.5, 5.5, 7.5, 9.5):
        ax.axvline(x, color="white", linewidth=2)
    ax.axvline(7.5, color="#a63b2e", linewidth=3)
    ax.set_title("Separate encoders: every initialization retained", pad=14)
    fig.text(
        0.5,
        0.03,
        (
            "Blue: K=4 and full ARI ≥0.8 (larger = more recovery).\n"
            "Orange: K>1 Gaussian returns (larger = more splitting). "
            "Counts /5 datasets; no automatic model choice."
        ),
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.22, right=0.99, top=0.9, bottom=0.23)
    out = repo / "docs/figures/separate-neural"
    out.mkdir(parents=True, exist_ok=False)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(out / ("architecture-recovery." + suffix), dpi=180)
    svg = out / "architecture-recovery.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    (out / "figure-data.json").write_text(
        json.dumps({"arms": arms, "cells": cells, "counts": values.tolist()}, indent=2) + "\n"
    )
    print(json.dumps({"passed": True, "groups": len(groups), "figure_cells": int(values.size)}))


if __name__ == "__main__":
    main()
