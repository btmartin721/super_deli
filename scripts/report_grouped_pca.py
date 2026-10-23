"""Export audited group-mean PCA comparisons and figures without model refits."""

import argparse
import shutil
from pathlib import Path

from report_shared_calibration import digest, read, save

ARMS = {
    "source_2": "Source rows · 2 joint PCs",
    "source_4": "Source rows · 4 joint PCs",
    "target_2": "Group means · 2 joint PCs",
    "target_4": "Group means · 4 joint PCs",
}
STRUCTURED = {
    "archived": "Archived · 64 groups",
    "fresh_64": "Fresh · 64 groups",
    "fresh_256": "Fresh · 256 groups",
    "unequal_variance": "Unequal variance · 64",
}
UNSTRUCTURED = {
    "independent_gaussian": "Independent Gaussian",
    "correlated_gaussian": "Correlated Gaussian",
    "cline": "Continuous cline",
}


def collect(root, output):
    summary = read(root / "audit/summary.json")
    hashes = read(root / "audit/files.json")
    if (
        summary["status"] != "complete"
        or summary["significance_fits"]
        or hashes != read(root / "study/files.json")
    ):
        raise ValueError("Expected a completed audited point/stability study")
    for relative, expected in hashes.items():
        if digest(root / "study" / relative) != expected:
            raise ValueError(f"Changed audited input: {relative}")
    output.mkdir(exist_ok=False)
    for name in ("points.csv", "candidates.csv", "paired.csv", "stability.csv"):
        shutil.copyfile(root / "audit" / name, output / name)
    for name in ("configuration.json", "grid.json", "source_study.json", "datasets.json"):
        save(output / name, read(root / "study" / name))
    for name in ("runtime_sources.json", "runtime_versions.json"):
        save(output / name, read(root / name))
    save(output / "summary.json", summary)
    save(output / "execution.json", read(root / "study-execution.json"))
    return summary


def stability_quality(root, output):
    """Post-fit truth agreement and pairwise support, separate from significance."""
    import numpy as np
    from report_shared_calibration import table

    from super_deli.benchmarking.recovery_analysis import _ari

    dataset_rows = read(root / "study/datasets.json")
    replicas, quality = [], []
    for dataset in dataset_rows:
        if dataset["cohort"] not in {"fresh_64", "fresh_256"}:
            continue
        base = root / "study" / dataset["cohort"] / f"dataset_{dataset['repetition']:04d}"
        truth = np.asarray(read(base / "truth.json")["labels"])
        for arm in ARMS:
            run = base / arm / "run"
            record = read(base / arm / "record.json")
            if not record["bootstrap"]["requested"] or record["status"] != "complete":
                continue
            present = []
            for row in read(run / "bootstrap_replicates.json"):
                result = {
                    "cohort": dataset["cohort"],
                    "repetition": dataset["repetition"],
                    "arm": arm,
                    "replicate": row["replicate"],
                    "status": row["status"],
                    "selected_k": row.get("selected_k"),
                    "n_oob": len(row["oob_indices"]),
                    "ari_all": None,
                    "ari_oob": None,
                }
                if row["status"] == "complete":
                    labels = np.asarray(row["labels"])
                    indices = np.asarray(row["oob_indices"])
                    result["ari_all"] = _ari(truth, labels)
                    result["ari_oob"] = (
                        _ari(truth[indices], labels[indices]) if len(indices) > 1 else None
                    )
                    present.append(result)
                replicas.append(result)
            result = {
                "cohort": dataset["cohort"],
                "repetition": dataset["repetition"],
                "arm": arm,
                "requested": record["bootstrap"]["requested"],
                "available": len(present),
            }
            for key in ("ari_all", "ari_oob"):
                values = [p[key] for p in present if p[key] is not None]
                result[f"mean_{key}"] = float(np.mean(values)) if values else None
                result[f"median_{key}"] = float(np.median(values)) if values else None
            lower = np.tril(np.ones((len(truth), len(truth)), dtype=bool), -1)
            same = truth[:, None] == truth[None, :]
            with np.load(run / "stability.npz", allow_pickle=False) as arrays:
                for pair_type, mask in (("within", lower & same), ("between", lower & ~same)):
                    oob = arrays["oob_coassignment"][mask]
                    finite = oob[np.isfinite(oob)]
                    result[f"mean_oob_coassignment_{pair_type}"] = (
                        float(finite.mean()) if len(finite) else None
                    )
                    result[f"oob_pairs_available_{pair_type}"] = len(finite)
                    result[f"oob_pairs_total_{pair_type}"] = int(mask.sum())
            quality.append(result)
    table(output / "stability_replicates.csv", replicas)
    table(output / "stability_quality.csv", quality)


def plot(summary, output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output.mkdir(exist_ok=False)
    plt.rcParams.update(
        {
            "font.size": 9,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "super-deli-grouped-pca-v1",
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    lookup = {(r["cohort"], r["arm"]): r for r in summary["summaries"]}
    colors = ("#0072B2", "#56B4E9", "#D55E00", "#009E73")
    for j, (arm, label) in enumerate(ARMS.items()):
        for panel, cohorts in ((0, STRUCTURED), (1, STRUCTURED), (2, UNSTRUCTURED)):
            for i, cohort in enumerate(cohorts):
                row = lookup[cohort, arm]
                y = i + (j - 1.5) * 0.17
                if panel == 1:
                    if row["ari"] is not None:
                        axes[panel].plot(
                            row["ari"]["median"], y, "o", color=colors[j], markersize=4
                        )
                else:
                    rate = row["rates"]["four" if panel == 0 else "non_single"]
                    mean = rate["rate"]
                    if mean is None:
                        axes[panel].text(
                            0.98, y, "unavailable", ha="right", color=colors[j], fontsize=7
                        )
                        continue
                    low, high = rate["wilson_95"]
                    axes[panel].errorbar(
                        mean,
                        y,
                        xerr=[[max(0, mean - low)], [max(0, high - mean)]],
                        fmt="o",
                        color=colors[j],
                        capsize=2,
                        markersize=4,
                        label=label if panel == 0 and i == 0 else None,
                    )
    for index, (cohorts, title, xlabel) in enumerate(
        (
            (STRUCTURED, "Generating group-number recovery", "K=4 / available observations"),
            (STRUCTURED, "Generating partition agreement", "Median adjusted Rand agreement"),
            (UNSTRUCTURED, "No discrete generating groups", "K>1 / available observations"),
        )
    ):
        axis = axes[index]
        axis.set(
            yticks=range(len(cohorts)),
            yticklabels=list(cohorts.values()),
            ylim=(len(cohorts) - 0.5, -0.5),
            xlim=(-0.035, 1.035),
            title=title,
            xlabel=xlabel,
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", alpha=0.15)
        if index != 1:
            axis.xaxis.set_major_formatter(PercentFormatter(1))
    figure.suptitle(
        "Grouped morphology + ecology: fixed PCA fit-level comparison", fontsize=13, y=0.99
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        fontsize=8,
    )
    figure.text(
        0.015,
        0.025,
        "25 observations/cohort; pointwise Wilson 95% intervals for selection rates; "
        "ARI medians are descriptive.\nNo new default or significance claim. No-group K "
        "selection is not a calibrated type-I error rate; stability is reported separately.",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.1, 1, 0.88), w_pad=2)
    for extension in ("png", "pdf", "svg"):
        path = output / f"grouped-pca.{extension}"
        metadata = (
            {"CreationDate": None, "ModDate": None}
            if extension == "pdf"
            else {"Date": None}
            if extension == "svg"
            else {}
        )
        figure.savefig(path, dpi=180, metadata=metadata)
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_dir, args.figure_dir):
        if any(
            output.resolve().is_relative_to((args.artifact_root / name).resolve())
            for name in ("study", "audit")
        ):
            raise ValueError("Reports must be outside immutable study/audit directories")
    summary = collect(args.artifact_root, args.output_dir)
    stability_quality(args.artifact_root, args.output_dir)
    plot(summary, args.figure_dir)
    save(
        args.output_dir / "manifest.json",
        {
            "local_artifact_id": "grouped-pca-20261023",
            "frozen_wheel_sha256": read(args.artifact_root / "runtime_package.json")["sha256"],
            "audit_summary_sha256": digest(args.artifact_root / "audit/summary.json"),
            "audited_inventory_sha256": digest(args.artifact_root / "audit/files.json"),
            "published_artifacts": {p.name: digest(p) for p in sorted(args.output_dir.iterdir())},
            "figures": {p.name: digest(p) for p in sorted(args.figure_dir.iterdir())},
        },
    )


if __name__ == "__main__":
    main()
