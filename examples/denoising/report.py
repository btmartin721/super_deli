"""Export small audited aggregates and static scientific figures only."""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from super_deli.benchmarking.denoising_study import ARMS, CASES, CONDITIONS
from super_deli.io.config import read_json, write_json


def publish(study, repository):
    summary = read_json(study / "aggregate.json")
    audit = read_json(study / "main-audit.json")
    assert audit["passed"] and audit["attempted_workflows"] == 2618
    public = repository / "examples/denoising/results"
    public.mkdir(exist_ok=False)
    write_json(summary, public / "summary.json")
    write_json(
        {
            "preflight": read_json(study / "preflight-audit.json"),
            "main": audit,
            "resource_gate": read_json(study / "storage-runtime-gate.json"),
        },
        public / "independent-audit.json",
    )
    rows = read_json(study / "point-assessments.json")
    geometry = read_json(study / "geometry-assessments.json")
    groups, shapes = [], []
    for case in CASES:
        for n in (64, 256):
            for condition, arms in [
                *((c, ARMS) for c in CONDITIONS),
                ("source", ("source_reference",)),
            ]:
                for arm in arms:
                    cell = [
                        r
                        for r in rows
                        if (r["case"], r["n"], r["condition"], r["arm"])
                        == (case, n, condition, arm)
                    ]
                    done = [r for r in cell if r["status"] == "complete"]
                    g = {
                        "case": case,
                        "n": n,
                        "condition": condition,
                        "arm": arm,
                        "requested": len(cell),
                        "resolved": len(done),
                        "failed": len(cell) - len(done),
                        "correct_k_count": sum(r["correct_k"] for r in done),
                        "k_counts": dict(Counter(str(r["selected_k"]) for r in done)),
                    }
                    for field in ("ari", "coverage", "noise_count", "splits", "merges"):
                        values = [r[field] for r in done]
                        g["mean_" + field] = float(np.mean(values)) if values else None
                        g["dataset_sd_" + field] = (
                            float(np.std(values, ddof=1)) if len(values) > 1 else None
                        )
                    groups.append(g)
                    measured = [
                        r
                        for r in geometry
                        if (r["case"], r["n"], r["condition"], r["arm"])
                        == (case, n, condition, arm)
                    ]
                    shape = {k: g[k] for k in ("case", "n", "condition", "arm", "resolved")}
                    shape["threshold_separable_count"] = (
                        sum(r["graph"].get("mutual_threshold_separable", False) for r in measured)
                        if case != "gaussian"
                        else None
                    )
                    shape["mean_source_neighbor_overlap"] = (
                        float(
                            np.mean(
                                [
                                    r["neighborhood"]["source_standard_neighbor_overlap"]
                                    for r in measured
                                ]
                            )
                        )
                        if measured
                        else None
                    )
                    if arm.startswith(("neural_", "denoising_")) and measured:
                        shape["mean_final_clean_mse"] = float(
                            np.mean([r["clean_initial_final_mse"][-1] for r in measured])
                        )
                    shapes.append(shape)
    write_json(groups, public / "all-seeds.json")
    write_json(shapes, public / "geometry.json")
    # Only six underlying datasets receive stability; retain that boundary.
    write_json(read_json(study / "stability-assessments.json"), public / "stability.json")
    figure_dir = repository / "docs/figures/denoising"
    figure_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    colors = ("#0072B2", "#D55E00", "#009E73")
    names = ("PCA/HDBSCAN", "Ordinary AE/HDBSCAN", "Denoising AE/HDBSCAN")
    primary = summary["primary_rates"]
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 10), sharey=True)
    for i, case in enumerate(CASES):
        for j, n in enumerate((64, 256)):
            ax = axes[i, j]
            for a, (arm, color, label) in enumerate(
                zip(("density", "neural_0", "denoising_0"), colors, names, strict=True)
            ):
                cell = [
                    next(
                        r
                        for r in primary
                        if (r["case"], r["n"], r["condition"], r["arm"]) == (case, n, c, arm)
                    )
                    for c in CONDITIONS
                ]
                y = np.array(
                    [
                        r["resolved_correct_k_rate"]
                        if r["resolved_correct_k_rate"] is not None
                        else np.nan
                        for r in cell
                    ]
                )
                bounds = np.array([r["simultaneous_95_interval"] for r in cell]).T
                ax.errorbar(
                    np.arange(2) + (a - 1) * 0.19,
                    y,
                    yerr=np.maximum(0, np.vstack([y - bounds[0], bounds[1] - y])),
                    fmt="o",
                    color=color,
                    capsize=3,
                    label=label,
                )
            ax.set_title(f"{case.replace('_', ' ').capitalize()}, N={n}")
            ax.set_xticks([0, 1], ["No extra feature noise", "Extra feature noise"])
            ax.set_ylim(-0.04, 1.04)
            ax.set_xlim(-0.5, 1.5)
            ax.grid(axis="y", alpha=0.2)
            if j == 0:
                ax.set_ylabel("Fraction selecting generating K")
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.suptitle("Fixed denoising objective: paired pilot comparison", fontsize=15, y=0.995)
    fig.text(
        0.5,
        0.012,
        "Ten independent datasets/cell; seed zero prespecified. "
        "Correct K is not assignment recovery.\n"
        "Simultaneous intervals cover 36 rates and 12 differences; "
        "other seeds remain in the summary.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.925))
    save(fig, figure_dir / "correct-k")
    write_json(
        {"primary_rates": primary, "all_seed_groups": groups, "geometry": shapes},
        figure_dir / "figure-data.json",
    )
    # The first frozen moons observation is fixed before outcomes; no best example.
    out = study / "main/moons_n256_0000"
    with np.load(out / "generating-reference.npz", allow_pickle=False) as f:
        truth = f["truth"].copy()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6))
    for i, condition in enumerate(CONDITIONS):
        for j, (arm, label) in enumerate(
            zip(("density", "neural_0", "denoising_0"), names, strict=True)
        ):
            ax = axes[i, j]
            path = out / condition / "point" / arm / "point.npz"
            if not path.exists():
                ax.text(0.5, 0.5, "Fit withheld/failed", ha="center", transform=ax.transAxes)
                continue
            with np.load(path, allow_pickle=False) as f:
                scores, labels = f["scores"].copy(), f["labels"].copy()
            for group, color in enumerate(colors[:2]):
                ix = truth == group
                ax.scatter(
                    scores[ix, 0],
                    scores[ix, 1],
                    s=11,
                    c=color,
                    alpha=0.7,
                    label=f"Generating group {group + 1}",
                )
            ax.set_title(
                f"{label}\nK={len(set(labels) - {-1})}, noise={sum(labels < 0)}/{len(labels)}"
            )
            ax.set_xlabel("Coordinate 1")
            ax.set_ylabel(f"{condition.capitalize()} features\nCoordinate 2")
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.96),
    )
    fig.suptitle("First frozen N=256 moons observation, seed zero", fontsize=14, y=0.995)
    fig.text(
        0.5,
        0.015,
        "Colors are generating labels inspected after fitting; "
        "they never enter training or selection.\n"
        "Paired conditions are repeated measurements. "
        "Axes differ across representations.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.91))
    save(fig, figure_dir / "moons-example")
    print(
        {"public_groups": len(groups), "geometry_groups": len(shapes), "public_point_data": False}
    )


def save(fig, stem):
    for extension in ("png", "pdf", "svg"):
        target = stem.with_suffix("." + extension)
        fig.savefig(target, dpi=180)
        if extension == "svg":
            target.write_text(
                "\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()
    publish(args.study_dir, args.repository)
