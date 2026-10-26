"""Export compact results and fixed scientific figures from audited local evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from super_deli.benchmarking.moons_config import ARMS, CASES


def read(path):
    return json.loads(path.read_text())


def summarize(root, output):
    audit = read(root / "audit/summary.json")
    if audit["status"] != "passed":
        raise ValueError("Audited study required")
    output.mkdir(exist_ok=False)

    def write(name, value):
        (output / name).write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )

    rows = [read(p) for p in sorted((root / "study").glob("*/summary.json"))]
    fresh = [r for r in rows if r["origin"] == "fresh"]
    paired = []
    mechanisms = []
    point_records = []
    for row in rows:
        if row["origin"] == "replay":
            continue
        for arm in ARMS:
            m = read(root / "study" / row["key"] / f"{arm}.json")
            if m["status"] != "complete":
                continue
            point_records.append((row, arm, m))
    for origin in ("archived", "fresh"):
        for case in CASES:
            for n in (64, 256):
                cell = [
                    r for r in rows if r["origin"] == origin and r["case"] == case and r["n"] == n
                ]
                if not cell:
                    continue
                for arm in ARMS:
                    keys = {row["key"] for row in cell}
                    records = [m for row, a, m in point_records if row["key"] in keys and a == arm]
                    positive = [m for m in records if m["hierarchy_status"] == "available"]
                    denominator = len(positive)
                    root_choice = sum(m["selection"]["root_selected"] for m in positive)
                    ratios = [m["graph"].get("mutual_separation_ratio") for m in positive]
                    jaccard = [
                        m.get("branch_reference", {}).get("best_disjoint_min_jaccard")
                        for m in positive
                    ]
                    separable = [
                        m for m in positive if m["graph"].get("mutual_threshold_separable")
                    ]
                    mechanisms.append(
                        {
                            "origin": origin,
                            "case": case,
                            "n": n,
                            "arm": arm,
                            "datasets": len(cell),
                            "available_hierarchies": denominator,
                            "root_selected": root_choice,
                            "density_threshold_separable": len(separable)
                            if case != "gaussian"
                            else None,
                            "root_selected_despite_separability": sum(
                                m["selection"]["root_selected"] for m in separable
                            )
                            if case != "gaussian"
                            else None,
                            "mean_mutual_separation_ratio": float(
                                np.mean([x for x in ratios if x is not None])
                            )
                            if any(x is not None for x in ratios)
                            else None,
                            "mean_best_disjoint_min_jaccard": float(
                                np.mean([x for x in jaccard if x is not None])
                            )
                            if any(x is not None for x in jaccard)
                            else None,
                            "mean_surviving_leaves": float(
                                np.mean([m["selection"]["surviving_leaves"] for m in positive])
                            )
                            if positive
                            else None,
                        }
                    )
                    if origin != "fresh" or arm == "baseline":
                        continue
                    for metric in ("correct_k", "ari", "assigned_fraction"):
                        differences = []
                        for r in cell:
                            base = r["arms"].get("baseline", {})
                            other = r["arms"].get(arm, {})
                            if (
                                base.get("status") != "complete"
                                or other.get("status") != "complete"
                            ):
                                continue
                            a, b = other["metrics"].get(metric), base["metrics"].get(metric)
                            if a is not None and b is not None:
                                differences.append(float(a) - float(b))
                        values = np.array(differences)
                        paired.append(
                            {
                                "case": case,
                                "n": n,
                                "arm": arm,
                                "metric": metric,
                                "direction": "arm minus baseline",
                                "independent_datasets": len(values),
                                "mean_difference": float(np.mean(values)) if len(values) else None,
                                "descriptive_se": float(
                                    np.std(values, ddof=1) / np.sqrt(len(values))
                                )
                                if len(values) > 1
                                else None,
                            }
                        )
    # A separate interval API checks the beta-quantile calculation in the auditor.
    for rate in audit["primary_rates"]:
        if rate["failed_or_unattempted"] == 0:
            ci = binomtest(rate["correct_k"], rate["datasets"]).proportion_ci(
                confidence_level=1 - 0.05 / 48, method="exact"
            )
            np.testing.assert_allclose(
                [ci.low, ci.high],
                [rate["simultaneous_95_lower"], rate["simultaneous_95_upper"]],
                atol=1e-10,
            )
    write("primary-rates.json", audit["primary_rates"])
    write("descriptive-metrics.json", audit["descriptive_metrics"])
    write("paired-differences.json", paired)
    write("mechanisms.json", mechanisms)
    fields = list(audit["primary_rates"][0])
    with (output / "primary-rates.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit["primary_rates"])
    main = read(root / "study/summary.json")
    pre = read(root / "preflight/summary.json")
    summary = {
        "schema_version": 1,
        "version": "0.49.0.dev0",
        "main": main,
        "preflight": pre,
        "archive_audit": {
            k: v
            for k, v in read(root / "archive-audit/summary.json").items()
            if k != "source_binding"
        },
        "independent_audit": {
            k: audit[k]
            for k in (
                "status",
                "audited_successful_fits",
                "verified_replays",
                "unavailable_hierarchies",
            )
        },
        "fresh_independent_datasets": len(fresh),
        "primary_rates": 48,
        "uncertainty": (
            "Exact simultaneous primary intervals; secondary paired means/SEs are descriptive"
        ),
        "preservation": "Production models, defaults and historical p-values unchanged",
        "limitations": [
            "Synthetic fixed two-dimensional laws",
            "Oracle branch/graph diagnostics use truth only after fitting",
            "Archived comparisons are exploratory",
            "No new calibration, resampling study or preferred default",
        ],
    }
    write("summary.json", summary)
    write(
        "checksums.json",
        {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.iterdir())
            if p.is_file()
        },
    )
    return summary


def figures(root, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    from super_deli.benchmarking.moons_archive import source_matrix

    destination.mkdir(exist_ok=False)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(destination / f"{name}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # Fixed original observation index zero at both sample sizes, never chosen by outcome.
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    for i, n in enumerate((64, 256)):
        key = f"archived_moons_n{n}_0000"
        p = root / "study" / key
        x, _ = source_matrix(p / "raw/traits.tsv")
        with np.load(p / "reference.npz", allow_pickle=False) as a:
            truth = a["truth"]
        with np.load(p / "baseline.npz", allow_pickle=False) as a:
            scores, labels = a["scores"], a["labels"]
        for j, (values, group, title) in enumerate(
            (
                (x, truth, "Raw traits: generating groups"),
                (scores, truth, "Standardized full PCA: generating groups"),
                (scores, labels, "Baseline density labels (arbitrary IDs)"),
            )
        ):
            ax = axes[i, j]
            for k in sorted(set(group)):
                mask = group == k
                ax.scatter(
                    values[mask, 0],
                    values[mask, 1],
                    s=15 if n == 64 else 8,
                    color="#9a9a9a" if k < 0 else plt.get_cmap("tab10")(int(k) % 10),
                    label="Unassigned/noise" if k < 0 else f"Group {k}",
                    alpha=0.8,
                )
            ax.set_title(f"N={n} | {title}", fontsize=9)
            ax.set_xlabel("Trait 1" if j == 0 else "PC1")
            ax.set_ylabel("Trait 2" if j == 0 else "PC2")
            ax.set_aspect("equal", adjustable="box")
            if j == 2:
                ax.legend(fontsize=8, loc="best")
        # Within each row, prediction and truth use identical PCA coordinate axes.
        for direction in ("x", "y"):
            limits = getattr(axes[i, 1], f"get_{direction}lim")()
            getattr(axes[i, 2], f"set_{direction}lim")(limits)
    fig.suptitle("Two retained PCs preserve standardized geometry; density labels can still merge")
    save(fig, "fixed-moons-examples")
    audit = read(root / "audit/summary.json")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    row_keys = [(c, n) for c in CASES for n in (64, 256)]
    rates = {(r["case"], r["n"], r["arm"]): r for r in audit["primary_rates"]}
    desc = {
        (r["case"], r["n"], r["arm"]): r
        for r in audit["descriptive_metrics"]
        if r["origin"] == "fresh"
    }
    for i, title in enumerate(("Correct generating group count", "Assigned coverage (mean)")):
        values = np.array(
            [
                [
                    rates[c, n, a]["rate_lower"]
                    if i == 0
                    else desc[c, n, a]["mean_assigned_fraction"]
                    for a in ARMS
                ]
                for c, n in row_keys
            ]
        )
        image = axes[i].imshow(values, norm=Normalize(0, 1), cmap="viridis", aspect="auto")
        for y, (c, n) in enumerate(row_keys):
            for x, arm in enumerate(ARMS):
                rate = rates[c, n, arm]
                text = (
                    f"{rate['correct_k']}/{rate['datasets']}"
                    if i == 0
                    else f"{100 * values[y, x]:.0f}%"
                )
                axes[i].text(
                    x,
                    y,
                    text,
                    ha="center",
                    va="center",
                    color="white" if values[y, x] < 0.5 else "black",
                    fontsize=8,
                )
        axes[i].set_xticks(
            range(len(ARMS)),
            [
                "Baseline",
                "No single",
                "Leaf",
                "Samples 2",
                "Samples 10",
                "Size 3",
                "Size 10",
                "Center only",
            ],
            fontsize=8,
        )
        axes[i].set_yticks(range(len(row_keys)), [f"{c}, N={n}" for c, n in row_keys], fontsize=8)
        axes[i].set_title(title)
        fig.colorbar(image, ax=axes[i], fraction=0.025, pad=0.02)
    fig.suptitle("Fresh frozen comparisons: group count and coverage must be read together")
    save(fig, "fresh-recovery-coverage")
    fig, axes = plt.subplots(
        2, 2, figsize=(9, 7), constrained_layout=True, sharex=True, sharey=True
    )
    for i, case in enumerate(("moons_008", "moons_002")):
        for j, n in enumerate((64, 256)):
            ax = axes[i, j]
            for p in sorted((root / "study").glob(f"fresh_{case}_n{n}_*/baseline.json")):
                m = read(p)
                if m.get("hierarchy_status") != "available":
                    continue
                t = m["selection"]
                den = t["root_stability"] + t["root_best_child_sum"]
                y = t["root_margin"] / den if den else 0
                k = m["selected_k"]
                ax.scatter(
                    m["graph"]["mutual_separation_ratio"],
                    y,
                    color="#0072b2" if k == 1 else "#d55e00" if k == 2 else "#009e73",
                    s=24,
                    alpha=0.8,
                )
            ax.axvline(1, color=".5", linestyle="--", linewidth=0.8)
            ax.axhline(0, color=".5", linestyle="--", linewidth=0.8)
            ax.set_title(f"{case}, N={n}")
            ax.set_xlabel("Between bridge / within bottleneck (oracle)")
            ax.set_ylabel("Normalized root minus best-child stability")
    from matplotlib.lines import Line2D

    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor=c, label=k)
            for c, k in [("#0072b2", "K=1"), ("#d55e00", "K=2"), ("#009e73", "Other K")]
        ],
        loc="outside lower center",
        ncol=3,
    )
    fig.suptitle("Density connectivity and branch selection are separate diagnostic questions")
    save(fig, "connectivity-selection")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path)
    args = parser.parse_args()
    result = summarize(args.artifact_root.resolve(), args.output_dir.resolve())
    if args.figure_dir:
        figures(args.artifact_root.resolve(), args.figure_dir.resolve())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
