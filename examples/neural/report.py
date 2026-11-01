"""Publish small audited aggregates and scientific figures, never fitted weights."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from super_deli.io.config import read_json

ARMS = ("gmm", "density", "neural_0")
NAMES = ("PCA + GMM", "PCA + HDBSCAN", "Autoencoder + HDBSCAN (seed 0)")
COLORS = ("#0072B2", "#E69F00", "#009E73")
CASES = ("gaussian", "four_spherical", "moons", "cline")
LABELS = ("Gaussian", "Four groups", "Curved moons", "Continuous cline")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def publish(root, repository):
    summary = read_json(root / "aggregate.json")
    audit = read_json(root / "main-audit.json")
    assert audit["passed"] and audit["attempted_workflows"] == 1210
    groups = {(g["case"], g["n"], g["arm"]): g for g in summary["groups"]}
    public = repository / "examples" / "neural" / "results"
    save_json(public / "summary.json", summary)
    save_json(
        public / "independent-audit.json",
        {
            "preflight": read_json(root / "preflight-audit.json"),
            "main": audit,
            "resource_gate": read_json(root / "storage-runtime-gate.json"),
            "software": read_json(root / "frozen.json")["software"],
            "source_sha256": read_json(root / "frozen.json")["source_sha256"],
        },
    )
    figure_dir = repository / "docs" / "figures" / "tabular-neural"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    figure_data = []
    width = 0.24
    for column, n in enumerate((64, 256)):
        top, bottom = axes[:, column]
        top.set_title(f"N = {n}: generating group count", fontweight="bold")
        bottom.set_title(f"N = {n}: assigned coverage", fontweight="bold")
        for a, (arm, name, color) in enumerate(zip(ARMS, NAMES, COLORS, strict=True)):
            x = np.arange(3) + (a - 1) * width
            cells = [groups[(c, n, arm)] for c in CASES[:3]]
            values = np.array([g["correct_k_rate_resolved"] for g in cells])
            bounds = np.array([g["simultaneous95_resolved"] for g in cells]).T
            top.bar(x, values, width=width, color=color, label=name)
            top.errorbar(
                x,
                values,
                yerr=np.vstack([values - bounds[0], bounds[1] - values]),
                fmt="none",
                ecolor="#222222",
                linewidth=0.8,
                capsize=2,
            )
            for p, g in zip(x, cells, strict=True):
                top.text(
                    p,
                    1.04,
                    f"{g['correct_k_count']}/{g['successful']}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
                figure_data.append(g)
            coverage = [groups[(c, n, arm)]["median_coverage"] for c in CASES]
            bottom.bar(np.arange(4) + (a - 1) * width, coverage, width=width, color=color)
        top.set_xticks(
            np.arange(3), ["Gaussian\nK = 1", "Four groups\nK = 4", "Curved moons\nK = 2"]
        )
        bottom.set_xticks(
            np.arange(4), ["Gaussian", "Four\ngroups", "Curved\nmoons", "Continuous\ncline"]
        )
        for ax in (top, bottom):
            ax.set_ylim(0, 1.16)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("Correct K among resolved fits")
    axes[1, 0].set_ylabel("Median fraction assigned")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.98)
    )
    fig.suptitle(
        "First neural representation compared with classical baselines", y=1.02, fontsize=15
    )
    fig.text(
        0.5,
        0.015,
        "Ten independent datasets per cell. Neural seed 0 was prespecified; "
        "other seeds are reported separately.\n"
        "Error bars: simultaneous exact 95% intervals for 18 rates. "
        "Training-cohort results, not species validation.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.93), h_pad=2)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(figure_dir / f"recovery.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    svg = figure_dir / "recovery.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    save_json(
        figure_dir / "figure-data.json",
        {
            "primary_rates": figure_data,
            "coverage": [g for g in summary["groups"] if g["arm"] in ARMS],
            "interval": "Bonferroni simultaneous exact 95% across 18 primary rates",
        },
    )
    lines = [
        "# First neural-model validation findings",
        "",
        "The completed checkpoint adds an opt-in CPU autoencoder/HDBSCAN workflow. "
        "It preserves the existing classical models, defaults, schemas and historical p-values. "
        "The new groups are transductive density hypotheses, with no species-significance claim.",
        "",
        "## Completed execution",
        "",
        f"The separate 120-workflow preflight passed with zero failures. The main study attempted "
        f"all 1,210 planned workflows across 80 fresh datasets, including paired unit subsamples "
        f"and ten exact point replays. The main audit records {audit['failed_workflows']} failed "
        "workflows; none were replaced. Reference labels were evaluated only after fitting. "
        "The numerical audit and full detailed evidence remain local. Only small aggregates "
        "and figures are published.",
        "",
        "## Group-count recovery and coverage",
        "",
        "Entries are correct-K selections / resolved datasets. Each cell requested ten datasets. "
        "Read failure bounds and the exact pointwise/simultaneous intervals in the public summary. "
        "Seed zero was chosen before generation, not selected as a winning neural initialization.",
        "",
        "| Law | N | PCA/GMM | PCA/HDBSCAN | Autoencoder/HDBSCAN, seed 0 |",
        "|---|---:|---:|---:|---:|",
    ]
    for case, name in zip(CASES[:3], LABELS[:3], strict=True):
        for n in (64, 256):
            cells = [groups[(case, n, a)] for a in ARMS]
            values = [f"{g['correct_k_count']}/{g['successful']}" for g in cells]
            lines.append(f"| {name} | {n} | " + " | ".join(values) + " |")
    lines += [
        "",
        "![Recovery and coverage](figures/tabular-neural/recovery.png)",
        "",
        "Correct K and accurate assignment are separate. Full-unit ARI treats each noise "
        "observation as a separate unresolved singleton; assigned-only ARI must be read with "
        "coverage. Gaussian extra-group selection is descriptive overpartitioning, not a "
        "calibrated significance-test false-positive rate. The continuous cline has no correct K.",
        "",
        "The neural workflow selected four groups in all ten N=64 four-group datasets at each "
        "of the three initialization seeds. At N=256 it selected four groups in eight of ten "
        "datasets at every seed, while PCA/GMM selected four in all ten. None of the neural "
        "seeds recovered the two moons groups. Gaussian K=1 selections also leave many "
        "observations unassigned: seed zero's median assigned coverage was 50.0% at N=64 "
        "and 75.4% at N=256. Seed 202 selected two groups in one N=256 Gaussian dataset. "
        "These results support a working neural alternative, without establishing improved "
        "performance across geometries or seeds.",
        "",
        "| Law | N | Arm | Median full-unit ARI | Median assigned coverage |",
        "|---|---:|---|---:|---:|",
    ]
    for case, name in zip(CASES, LABELS, strict=True):
        for n in (64, 256):
            for arm, label in zip(ARMS, NAMES, strict=True):
                g = groups[(case, n, arm)]
                agreement = "not defined" if g["median_ari"] is None else f"{g['median_ari']:.3f}"
                coverage = (
                    "unavailable"
                    if g["median_coverage"] is None
                    else f"{100 * g['median_coverage']:.1f}%"
                )
                lines.append(f"| {name} | {n} | {label} | {agreement} | {coverage} |")
    lines += [
        "",
        "## Initialization and conditional subsampling",
        "",
        "The public summary retains seeds 0, 101 and 202 rather than promoting the best one. "
        "Initialization agreement is conditional on jointly non-noise units, with its coverage. "
        "A degenerate constant partition can have ARI one "
        "without demonstrating recovered structure.",
        "",
        "Twenty shared 80% whole-unit subsamples were fitted on the first observation in each "
        "of eight cells, for every arm. That is one stability example per cell, "
        "not ten independent "
        "stability datasets. Every draw refits raw preprocessing, representation and clustering. "
        "Native coassignment uses successful pair inclusions, keeps failures and bounds, and "
        "never assigns absent observations. The within-point statistic is pair-inclusion-weighted "
        "coassignment among original positive-cluster pairs; it is not species support or "
        "external predictive accuracy. The GMM study subsamples "
        "do not replace its ordinary bootstrap.",
        "",
        "## Interpretation limits",
        "",
        "This is a small, frozen pilot with ten independent datasets per cell and one "
        "architecture/training budget. Reconstruction is in-cohort training MSE, not an external "
        "prediction score. Neural latent standardization differs from native PCA-score scaling, "
        "so the experiment compares whole workflows. No configuration, seed or covariance family "
        "is automatically selected or promoted. Missingness, genomic/shape/multimodal neural "
        "adapters, anatomical landmark learning and significance calibration remain later work. "
        "No SPARUS or other empirical fitting was performed and nobody was contacted.",
        "",
        "See the [model and API](tabular-neural.md), [fixed protocol](tabular-neural-study.md), "
        "[small results](../examples/neural/results/summary.json) and "
        "[independent audit](../examples/neural/results/independent-audit.json).",
        "",
    ]
    with (repository / "docs" / "tabular-neural-results.md").open("x") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-dir", type=Path, required=True)
    p.add_argument("--repository", type=Path, required=True)
    args = p.parse_args()
    publish(args.study_dir, args.repository)
