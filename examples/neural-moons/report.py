"""Export small audited summaries and figures; detailed coordinates stay local."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from super_deli.benchmarking.neural_moons_config import POLICIES, REPRESENTATIONS
from super_deli.io.config import read_json, write_json

LABELS = (
    "Generating 2D (oracle)",
    "Observed traits 8D",
    "PCA 2D",
    "AE seed 0: raw",
    "AE seed 101: raw",
    "AE seed 202: raw",
    "AE seed 0: standardized",
    "AE seed 101: standardized",
    "AE seed 202: standardized",
)
COLORS = ("#0072B2", "#D55E00", "#009E73")
POLICY_NAMES = ("Unchanged EOM", "EOM: root excluded", "Leaf selection")


def publish(study, repository):
    summary = read_json(study / "aggregate.json")
    audit = read_json(study / "main-audit.json")
    assert not summary["fixture"] and audit["passed"]
    assert audit["attempted"] == 548 and audit["selection_evaluations"] == 1628
    public = repository / "examples/neural-moons/results"
    public.mkdir(parents=True, exist_ok=True)
    write_json(summary, public / "summary.json")
    write_json(
        {
            "archive": {
                k: v
                for k, v in read_json(study / "archive-binding.json").items()
                if k
                in {
                    "kind",
                    "passed",
                    "original_workflows",
                    "archive_failures",
                    "source_software",
                    "source_bundle_verified",
                }
            },
            "preflight": read_json(study / "preflight-audit.json"),
            "main": audit,
            "resource_gate": read_json(study / "storage-runtime-gate.json"),
        },
        public / "independent-audit.json",
    )
    raw_geometry = read_json(study / "geometry-assessments.json")
    geometry = []
    for case in ("gaussian", "four_spherical", "moons"):
        for n in (64, 256):
            for space in REPRESENTATIONS:
                cell = [
                    r
                    for r in raw_geometry
                    if r["case"] == case and r["n"] == n and r["representation"] == space
                ]
                row = {
                    "case": case,
                    "n": n,
                    "representation": space,
                    "successful": len(cell),
                    "threshold_separable_count": sum(
                        r["graph"].get("mutual_threshold_separable", False) for r in cell
                    )
                    if case != "gaussian"
                    else None,
                }
                for metric in (
                    "source_standard_neighbor_overlap",
                    "traits_standard_neighbor_overlap",
                    "cross_neighbor_fraction",
                ):
                    row["mean_" + metric] = (
                        float(np.mean([r["neighborhood"][metric] for r in cell])) if cell else None
                    )
                vals = [r["branch_reference"].get("best_disjoint_min_jaccard") for r in cell]
                vals = [v for v in vals if v is not None]
                row["mean_best_disjoint_branch_jaccard"] = float(np.mean(vals)) if vals else None
                geometry.append(row)
    write_json(geometry, public / "geometry.json")
    figure_dir = repository / "docs/figures/neural-moons"
    figure_dir.mkdir(parents=True, exist_ok=True)
    groups = {(g["case"], g["n"], g["representation"], g["policy"]): g for g in summary["groups"]}
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.6), sharey=True)
    y = np.arange(9)
    for col, n in enumerate((64, 256)):
        for p, (policy, color, name) in enumerate(zip(POLICIES, COLORS, POLICY_NAMES, strict=True)):
            cell = [groups[("moons", n, r, policy)] for r in REPRESENTATIONS]
            position = y + (p - 1) * 0.22
            axes[0, col].scatter(
                [g["correct_k_count"] for g in cell], position, color=color, s=28, label=name
            )
            axes[1, col].scatter([g["mean_ari"] for g in cell], position, color=color, s=28)
        axes[0, col].set_title(f"N={n}: two-group count", fontweight="bold")
        axes[1, col].set_title(f"N={n}: assignment agreement", fontweight="bold")
        axes[0, col].set_xlim(-0.5, 10.5)
        axes[0, col].set_xticks([0, 2, 4, 6, 8, 10])
        axes[0, col].set_xlabel("Datasets selecting K=2 / 10")
        axes[1, col].set_xlim(-0.05, 1.05)
        axes[1, col].set_xlabel("Mean full-unit adjusted Rand agreement")
    for ax in axes.flat:
        ax.set_yticks(y, LABELS)
        ax.set_ylim(8.7, -0.7)
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, names, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.61, 0.98)
    )
    fig.suptitle("Removing the root does not consistently recover the moons", fontsize=15, y=1.015)
    fig.text(
        0.6,
        0.015,
        "Retrospective: ten archived datasets per cell. Seeds are repeated measurements.\n"
        "Count and assignment agreement are separate; noise is an unresolved singleton.\n"
        "Pointwise descriptive intervals and failure bounds are retained in the public summary.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.085, 1, 0.94), h_pad=2)
    save_figure(fig, figure_dir / "moons-recovery")
    fig, axes = plt.subplots(1, 3, figsize=(13, 6.7), sharey=True)
    for n, offset, marker, color in ((64, -0.12, "o", COLORS[0]), (256, 0.12, "s", COLORS[1])):
        cell = [
            next(
                g
                for g in geometry
                if g["case"] == "moons" and g["n"] == n and g["representation"] == r
            )
            for r in REPRESENTATIONS
        ]
        values = (
            [g["threshold_separable_count"] for g in cell],
            [g["mean_best_disjoint_branch_jaccard"] for g in cell],
            [g["mean_source_standard_neighbor_overlap"] for g in cell],
        )
        for ax, v in zip(axes, values, strict=True):
            ax.scatter(v, y + offset, color=color, marker=marker, s=35, label=f"N={n}")
    for ax in axes:
        ax.set_yticks(y, LABELS)
        ax.set_ylim(8.5, -0.5)
        ax.grid(axis="x", alpha=0.18)
    axes[0].set_title("Complete reference-group cut", fontweight="bold")
    axes[0].set_xlabel("Threshold-separable datasets / 10")
    axes[0].set_xlim(-0.5, 10.5)
    axes[1].set_title("Structure in surviving branches", fontweight="bold")
    axes[1].set_xlabel("Mean best disjoint branch Jaccard")
    axes[1].set_xlim(-0.03, 1.03)
    axes[2].set_title("Generating-neighbor retention", fontweight="bold")
    axes[2].set_xlabel("Mean four-neighbor overlap")
    axes[2].set_xlim(-0.03, 1.03)
    axes[2].legend(frameon=False, loc="lower right")
    fig.suptitle(
        "Geometric degradation precedes the neural encoder in this benchmark", fontsize=14, y=0.99
    )
    fig.text(
        0.6,
        0.025,
        "Oracle reference diagnostics evaluated after fitting; not deployable species rules.\n"
        "Dimension, metric, feature mapping and added measurement noise change together.\n"
        "The contrasts do not identify measurement noise alone as the cause.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.93), w_pad=1.4)
    save_figure(fig, figure_dir / "geometry")
    write_json(
        {
            "selection_groups": summary["groups"],
            "geometry": geometry,
            "scope": "Retrospective pilot; no fresh performance, species or default claim",
        },
        figure_dir / "figure-data.json",
    )


def save_figure(fig, stem):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix("." + extension), dpi=170, bbox_inches="tight")
    plt.close(fig)
    svg = stem.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--repository", required=True, type=Path)
    args = parser.parse_args()
    publish(args.study_dir, args.repository)
