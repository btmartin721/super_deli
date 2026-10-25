"""Summarize audited locality stability without treating draws as independent datasets."""

import argparse
import csv
import hashlib
import io
import json
import statistics
import tarfile
from pathlib import Path


def describe(values):
    values = [float(v) for v in values if v is not None]
    return (
        dict(
            mean=statistics.mean(values),
            median=statistics.median(values),
            minimum=min(values),
            maximum=max(values),
        )
        if values
        else None
    )


def report(root, output, figures):
    import numpy as np

    source = Path(root)
    output = Path(output)
    figures = Path(figures)
    output.mkdir(parents=True, exist_ok=False)
    figures.mkdir(parents=True, exist_ok=False)
    ledger = [json.loads(s) for s in (source / "study/observations.jsonl").read_text().splitlines()]
    assert len(ledger) == 40
    assert json.loads((source / "audit/summary.json").read_text())["audited_searches"] == 24000
    assert json.loads((source / "stdlib.json").read_text())["audited_searches"] == 24000
    assert json.loads((source / "replays/summary.json").read_text())["verified_replays"] == 15
    rows = []
    for outer in sorted(ledger, key=lambda r: (r["cell"], r["repetition"])):
        names = {
            "workflow/summary.json",
            "workflow/stability.npz",
            "truth_evaluation.json",
            "workflow/point/selected_memberships.npz",
        }
        files = {}
        with tarfile.open(source / "study" / outer["archive"], "r|gz") as archive:
            for member in archive:
                if member.name in names:
                    files[member.name] = archive.extractfile(member).read()
        result = json.loads(files["workflow/summary.json"])
        truth = json.loads(files["truth_evaluation.json"])
        with np.load(io.BytesIO(files["workflow/point/selected_memberships.npz"])) as p:
            point = p["labels"]
        same = point[:, None] == point[None, :]
        off = ~np.eye(len(point), dtype=bool)
        within = same & off
        cross = (~same) & off
        success = result["stability"]["successful"]
        requested = result["stability"]["requested"]
        k = outer["point_selection"]["screened_k"]
        counts = {int(a): b for a, b in result["stability"]["selected_k_counts"].items()}
        row = dict(
            cell=outer["cell"],
            repetition=outer["repetition"],
            law=outer["cell_config"]["law"],
            role=outer["cell_config"]["role"],
            n=outer["cell_config"]["n_units"],
            point_k=k,
            point_ari=outer["point_ari"],
            requested_draws=requested,
            successful_draws=success,
            failed_draws=requested - success,
            conditional_same_k_frequency=counts.get(k, 0) / success if success else None,
            conditional_mean_draw_ari=describe([v["ari"] for v in truth])["mean"]
            if any(v["ari"] is not None for v in truth)
            else None,
        )
        for a in range(1, 5):
            row[f"selected_k{a}_count"] = counts.get(a, 0)

        def average(values, mask):
            return float(np.nanmean(values[mask])) if np.isfinite(values[mask]).any() else None

        with np.load(io.BytesIO(files["workflow/stability.npz"]), allow_pickle=False) as p:
            for prefix in ("all", "oob"):
                freq = p[prefix + "_conditional_frequency"]
                covered = np.isfinite(freq) & off
                agreement = np.where(same, freq, 1 - freq)
                row[prefix + "_covered_pair_fraction"] = int(covered.sum()) / int(off.sum())
                row[prefix + "_conditional_point_partition_agreement"] = average(agreement, covered)
                row[prefix + "_conditional_within_cohesion"] = average(freq, within)
                row[prefix + "_conditional_cross_merging"] = average(freq, cross)
            row["conditional_mean_exact_point_cluster_recovery"] = float(
                np.mean(p["exact_cluster_conditional_frequency"])
            )
            row["oob_min_successful_pairs"] = int(p["oob_successful"][off].min())
            row["oob_median_successful_pairs"] = float(np.median(p["oob_successful"][off]))
            widths = p["oob_conditional_mc_high"] - p["oob_conditional_mc_low"]
            row["oob_median_conditional_mc_interval_width"] = (
                float(np.nanmedian(widths[off])) if np.isfinite(widths[off]).any() else None
            )
        rows.append(row)
    with (output / "datasets.csv").open("x", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cells = []
    for name in dict.fromkeys(r["cell"] for r in rows):
        items = [r for r in rows if r["cell"] == name]
        cells.append(
            dict(
                cell=name,
                law=items[0]["law"],
                role=items[0]["role"],
                n=items[0]["n"],
                independent_datasets=len(items),
                point_k_counts={str(k): sum(r["point_k"] == k for r in items) for k in range(1, 5)},
                requested_draws=sum(r["requested_draws"] for r in items),
                successful_draws=sum(r["successful_draws"] for r in items),
                descriptive_dataset_summaries={
                    key: describe([r[key] for r in items])
                    for key in (
                        "point_ari",
                        "conditional_same_k_frequency",
                        "conditional_mean_draw_ari",
                        "conditional_mean_exact_point_cluster_recovery",
                        "all_conditional_within_cohesion",
                        "all_conditional_cross_merging",
                        "oob_conditional_within_cohesion",
                        "oob_conditional_cross_merging",
                        "oob_covered_pair_fraction",
                        "oob_median_successful_pairs",
                        "oob_median_conditional_mc_interval_width",
                    )
                },
                mean_dataset_conditional_selected_k_frequencies={
                    str(k): statistics.mean(
                        r[f"selected_k{k}_count"] / r["successful_draws"] for r in items
                    )
                    for k in range(1, 5)
                },
            )
        )
    study = json.loads((source / "study/summary.json").read_text())
    replay = json.loads((source / "replays/summary.json").read_text())
    storage = json.loads((source / "storage-preflight.json").read_text())
    summary = dict(
        cells=cells,
        main_study=study,
        replays=replay,
        preflight=dict(
            searches=4800,
            draws=792,
            both_audits="passed",
            projected_GiB=storage["projected_complete_bytes"] / 1024**3,
            cap_GiB=4,
        ),
        scientific_interpretation=(
            "Five independent datasets per cell; draws are repeated measurements, "
            "not independent datasets. No stability-probability calibration, "
            "species probability, significance or general performance guarantee."
        ),
        defaults_and_historical_pvalues_unchanged=True,
        detailed_artifacts_uploaded=False,
    )
    with (output / "summary.json").open("x") as h:
        json.dump(summary, h, indent=2, sort_keys=True)
        h.write("\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 9})
    fig, axes = plt.subplots(
        3, 1, figsize=(11, 10), gridspec_kw={"height_ratios": [2, 1, 1]}, layout="constrained"
    )
    colors = ["#7a7a7a", "#e69f00", "#56b4e9", "#009e73"]
    left = np.zeros(len(rows))
    for k, color in zip(range(1, 5), colors, strict=True):
        vals = np.array([r[f"selected_k{k}_count"] / r["successful_draws"] for r in rows])
        axes[0].bar(
            np.arange(len(rows)), vals, bottom=left, width=0.85, color=color, label=f"K={k}"
        )
        left += vals
    axes[0].set(
        ylabel="Conditional bootstrap frequency",
        ylim=(0, 1.06),
        title="Selected K across 99 whole-locality draws per dataset",
    )
    axes[0].legend(ncols=4, loc="upper right", frameon=False)
    axes[1].plot(
        [r["all_conditional_within_cohesion"] for r in rows],
        "o",
        ms=4,
        label="Within point-cluster cohesion",
        color="#0072b2",
    )
    axes[1].plot(
        [
            r["all_conditional_cross_merging"]
            if r["all_conditional_cross_merging"] is not None
            else np.nan
            for r in rows
        ],
        "x",
        ms=4,
        label="Across point-cluster merging",
        color="#d55e00",
    )
    axes[1].set(ylabel="All-unit pair frequency", ylim=(-0.05, 1.05))
    axes[1].legend(frameon=False, ncols=2)
    axes[2].plot([r["oob_covered_pair_fraction"] for r in rows], "o", ms=4, color="#6f3c8f")
    axes[2].set(
        ylabel="Both-OOB pair coverage",
        ylim=(0, 1.05),
        xlabel="Independent datasets; five per cell",
    )
    for ax in axes:
        for cut in range(5, len(rows), 5):
            ax.axvline(cut - 0.5, color=".85", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-1, len(rows))
        ax.set_xticks([])
    labels = [name.replace("_", " ") for name in dict.fromkeys(r["cell"] for r in rows)]
    axes[2].set_xticks(np.arange(2, len(rows), 5), labels, rotation=30, ha="right", fontsize=8)
    fig.suptitle(
        "Shared predictive-screened AIC: conditional resampling stability\n"
        "Fixed original folds and optimizer seeds; no species-probability interpretation",
        fontsize=12,
    )
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(figures / ("stability." + suffix), dpi=180)
    plt.close(fig)
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()
    }
    with (output / "checksums.json").open("x") as h:
        json.dump(files, h, indent=2, sort_keys=True)
        h.write("\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("artifacts", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--figure-dir", type=Path, required=True)
    a = p.parse_args()
    report(a.artifacts, a.output_dir, a.figure_dir)
