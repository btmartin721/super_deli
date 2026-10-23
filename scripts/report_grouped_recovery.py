"""Publish audited grouped recovery diagnostics without refitting models."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from report_shared_calibration import digest, read, save

COHORTS = {
    "archived": "Archived · 64 groups",
    "fresh_64": "Fresh · 64 groups",
    "fresh_256": "Fresh · 256 groups",
}
VARIANTS = (
    "baseline",
    "pcs_3",
    "pcs_4",
    "ecology_0.5",
    "ecology_2",
    "ecology_4",
    "starts_100",
    "views_4_joint_2",
    "views_4_joint_4",
    "first_specimen",
)


def collect(root, output):
    summary = read(root / "audit/summary.json")
    if summary["status"] != "complete" or summary["bootstrap_or_calibration_fits"]:
        raise ValueError("Expected completed point-only audit")
    hashes = read(root / "audit/files.json")
    if hashes != read(root / "study/files.json"):
        raise ValueError("Audit and study inventories disagree")
    for relative, expected in hashes.items():
        if digest(root / "study" / relative) != expected:
            raise ValueError(f"Changed audited input: {relative}")
    output.mkdir(exist_ok=False)
    for name in ("points.csv", "candidates.csv", "truth_initialized.csv"):
        shutil.copyfile(root / "audit" / name, output / name)
    for name in ("configuration.json", "grid.json", "source_study.json", "datasets.json"):
        save(output / name, read(root / "study" / name))
    for name in ("runtime_sources.json", "runtime_versions.json"):
        save(output / name, read(root / name))
    save(output / "summary.json", summary)
    save(output / "execution.json", read(root / "study-execution.json"))
    if (root / "arithmetic-audit.json").exists():
        save(output / "arithmetic-audit.json", read(root / "arithmetic-audit.json"))
    return summary


def plot(summary, output):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    output.mkdir(exist_ok=False)
    plt.rcParams.update(
        {
            "font.size": 9,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "super-deli-grouped-recovery-v1",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.3, 1]})
    lookup = {(r["cohort"], r["variant"]): r for r in summary["summaries"]}
    colors = ("#0072B2", "#D55E00", "#009E73")
    for j, (cohort, label) in enumerate(COHORTS.items()):
        for i, name in enumerate(VARIANTS):
            row = lookup[cohort, name]
            value, interval = row["selected_four_rate"], row["selected_four_wilson_95"]
            if value is None:
                axes[0].text(
                    0.98, i + (j - 1) * 0.2, "unavailable*", color=colors[j], ha="right", fontsize=7
                )
            if value is not None:
                lo, hi = interval
                axes[0].errorbar(
                    value,
                    i + (j - 1) * 0.2,
                    xerr=[[max(0, value - lo)], [max(0, hi - value)]],
                    fmt="o",
                    color=colors[j],
                    capsize=2,
                    markersize=4,
                    label=label if i == 0 else None,
                )
    axes[0].set(
        yticks=np.arange(10),
        yticklabels=[
            "Baseline",
            "3 joint PCs",
            "4 joint PCs",
            "Ecology weight 0.5",
            "Ecology weight 2",
            "Ecology weight 4",
            "100 EM starts",
            "4 PCs/view; 2 joint PCs",
            "4 PCs/view; 4 joint PCs",
            "First specimen/group",
        ],
        ylim=(9.6, -0.6),
        xlim=(-0.03, 1.03),
        xlabel="Selected K=4 / available observations",
        title="Group-number recovery · pointwise Wilson 95%",
    )
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(frameon=False, loc="upper right", fontsize=8)
    stages = [
        ("morphological_raw_group_a_r2", "Morphology raw means · a"),
        ("morphological_profile_a_r2", "Morphology group PCs · a"),
        ("ecological_raw_group_b_r2", "Ecology raw · b"),
        ("ecological_profile_b_r2", "Ecology PCs · b"),
        ("fused_b_r2", "Fused profiles · b"),
        ("joint_b_r2", "Joint PCs · b"),
    ]
    for j, (cohort, label) in enumerate(COHORTS.items()):
        row = lookup[cohort, "baseline"]
        axes[1].plot(
            [row[key]["median"] for key, _ in stages],
            np.arange(len(stages)),
            "o",
            color=colors[j],
            label=label,
            markersize=5,
            alpha=0.85,
        )
    axes[1].set(
        yticks=np.arange(len(stages)),
        yticklabels=[label for _, label in stages],
        ylim=(len(stages) - 0.5, -0.5),
        xlim=(0, 1.02),
        xlabel="Median generating-axis linear R-squared",
        title="Baseline signal retention · descriptive",
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", alpha=0.15)
    figure.suptitle("Grouped morphology + ecology: frozen recovery diagnosis", fontsize=13)
    figure.text(
        0.02,
        0.025,
        "25 observations/cohort; one constructed law; exploratory settings, no automatic "
        "default change.\n*Both four-PC/view arms unavailable at N=256 (25 failures each). "
        "First-specimen control also removes replication. K=4 selection is distinct from "
        "partition accuracy and species evidence.",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 0.95), w_pad=3)
    for extension in ("png", "pdf", "svg"):
        path = output / f"grouped-recovery.{extension}"
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
    plot(summary, args.figure_dir)
    save(
        args.output_dir / "manifest.json",
        {
            "local_artifact_id": "grouped-recovery-20261023",
            "frozen_wheel_sha256": read(args.artifact_root / "runtime_package.json")["sha256"],
            "audit_summary_sha256": digest(args.artifact_root / "audit/summary.json"),
            "audited_inventory_sha256": digest(args.artifact_root / "audit/files.json"),
            "published_artifacts": {p.name: digest(p) for p in sorted(args.output_dir.iterdir())},
            "figures": {p.name: digest(p) for p in sorted(args.figure_dir.iterdir())},
        },
    )


if __name__ == "__main__":
    main()
