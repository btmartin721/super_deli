"""Export portable grouped shared-family calibration evidence without model refits."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from report_shared_calibration import digest, read, save, table

NAMES = {
    "independent_gaussian": "Independent Gaussian · null",
    "correlated_gaussian": "Correlated Gaussian · null",
    "complementary": "Complementary four groups · alternative",
    "heavy_tailed": "Heavy-tailed t3 · stress",
    "zero_conditional_gaussian": "Zero conditional covariance · null",
    "rank_two_gaussian": "Rank-two conditional covariance · null",
    "weak_conditional_gaussian": "Weak full-rank covariance · null",
}


def collect(root, output):
    """Use audited relative paths and retain every outcome, including withheld tests."""
    summary = read(root / "study-audit/summary.json")
    hashes = read(root / "study-audit/files.json")
    calibration = summary["study"]["calibration"]
    if calibration["schema_version"] != 2 or calibration["model"].get("covariance_type") != "tied":
        raise ValueError("Expected a prespecified schema-2 shared-covariance study")

    def verified(relative):
        path = root / "study" / relative
        if digest(path) != hashes.get(relative):
            raise ValueError(f"Changed audited input: {relative}")
        return path

    output.mkdir(exist_ok=False)
    records, diagnostics, sizes, ranks = [], [], {}, {}
    for case in NAMES:
        shard = read(verified(f"{case}/summary.json"))
        counts, rank_counts = Counter(), Counter()
        for row in shard["datasets"]:
            prefix = f"{case}/{row['directory']}/calibration"
            diagnostic = {
                key: row[key]
                for key in (
                    "case",
                    "repetition",
                    "statistic",
                    "p_value",
                    "reject",
                    "status",
                    "selected_k",
                )
            }
            layout = f"{prefix}/layout.json"
            diagnostic["null_fit_available"] = layout in hashes
            if layout in hashes:
                fitted = read(verified(layout))
                fit = fitted["diagnostics"]
                sampling = fitted["counts"]
                rank_counts[str(fit["conditional_rank"])] += 1
                diagnostic.update(
                    n_groups=len(sampling),
                    n_specimens=sum(sampling),
                    n_ecology_rows=len(
                        next(v for v in fitted["views"] if v["kind"] == "ecological")["unit_ids"]
                    ),
                    conditional_rank=fit["conditional_rank"],
                    boundary_or_near_boundary=fit["boundary_or_near_boundary"],
                    optimality_maximum=fit["optimality"]["maximum"],
                    negative_log_likelihood=fit["negative_log_likelihood"],
                    selected_start=fit["selected_start"],
                    accepted_starts=sum(a["accepted"] for a in fit["attempts"]),
                    declared_starts=len(fit["attempts"]),
                    **{f"groups_with_{n}_specimens": sampling.count(n) for n in range(1, 5)},
                )
            if row["outcome_file"] == "summary.json":
                result = read(verified(f"{prefix}/summary.json"))
                inference = result["inference"]
                interval = inference.get("conditional_null_tail_interval_95")
                failed_bounds = inference.get("failed_outcome_bounds")
                diagnostic.update(
                    failed_p_lower=failed_bounds[0] if failed_bounds else None,
                    failed_p_upper=failed_bounds[1] if failed_bounds else None,
                    exceedances=inference.get("exceedances"),
                    successful_draws=inference["successful"],
                    failed_draws=inference["failed"],
                    tail_lower_95=interval[0] if interval else None,
                    tail_upper_95=interval[1] if interval else None,
                )
            diagnostics.append(diagnostic)
            records.append(row)
            if row["selected_k"] is not None:
                counts[str(row["selected_k"])] += 1
        sizes[case] = {str(k): counts[str(k)] for k in range(1, calibration["model"]["max_k"] + 1)}
        ranks[case] = {str(k): rank_counts[str(k)] for k in range(5)}
    table(output / "datasets.csv", records)
    table(output / "diagnostics.csv", diagnostics)
    save(
        output / "summary.json",
        {**summary, "selected_k_counts": sizes, "conditional_rank_counts": ranks},
    )
    for name in ("runtime_sources.json", "runtime_versions.json", "study.json"):
        save(output / name, read(root / name))
    save(output / "execution.json", read(root / "study-execution.json"))
    return summary


def plot(summary, output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output.mkdir(exist_ok=False)
    plt.rcParams.update(
        {
            "font.size": 10,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "super-deli-shared-grouped-v1",
        }
    )
    figure, axis = plt.subplots(figsize=(10, 6))
    rates = {row["case"]: row for row in summary["rates"]}
    for i, case in enumerate(NAMES):
        rate = rates[case]
        if rate["mean"] is None:
            axis.text(0.03, i, "No available tests")
            continue
        mean, (low, high) = rate["mean"], rate["wilson_95"]
        color = (
            "#D55E00"
            if case == "heavy_tailed"
            else "#009E73"
            if case == "complementary"
            else "#0072B2"
        )
        axis.errorbar(
            mean,
            i,
            xerr=[[max(0, mean - low)], [max(0, high - mean)]],
            fmt="o",
            capsize=4,
            color=color,
            markersize=6,
        )
        axis.annotate(
            f"{rate['events']}/{rate['n_available']} rejected; "
            f"{rate['n_available']}/{summary['study']['n_datasets']} available",
            (mean, i),
            xytext=(-6 if mean > 0.75 else 6, 9),
            textcoords="offset points",
            ha="right" if mean > 0.75 else "left",
            fontsize=8,
        )
    axis.axvline(
        summary["study"]["calibration"]["alpha"],
        color=".5",
        linestyle="--",
        linewidth=1,
        label="Declared alpha = 5%",
    )
    axis.set(
        title="Shared-covariance grouped fitted-null calibration",
        xlabel="Rejected / available independent observations",
        xlim=(-0.02, 1.02),
        ylim=(6.65, -0.65),
        yticks=range(7),
        yticklabels=list(NAMES.values()),
    )
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower right", frameon=False, fontsize=8)
    figure.text(
        0.02,
        0.025,
        "Pointwise Wilson 95% intervals conditional on availability; "
        f"{summary['study']['n_datasets']} observations/scenario. "
        f"{summary['study']['calibration']['n_simulations']} draws/test.\n"
        "Boundary p-values are nonregular and approximate; "
        "no general size or species-boundary guarantee.",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.085, 1, 1))
    for extension in ("png", "pdf", "svg"):
        path = output / f"shared-grouped-calibration.{extension}"
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
            for name in ("study", "study-audit")
        ):
            raise ValueError("Reports must be outside immutable study/audit directories")
    summary = collect(args.artifact_root, args.output_dir)
    plot(summary, args.figure_dir)
    save(
        args.output_dir / "manifest.json",
        {
            "local_artifact_id": "shared-grouped-null-20261022",
            "frozen_wheel_sha256": read(args.artifact_root / "runtime_package.json")["sha256"],
            "audit_summary_sha256": digest(args.artifact_root / "study-audit/summary.json"),
            "audited_inventory_sha256": digest(args.artifact_root / "study-audit/files.json"),
            "published_artifacts": {p.name: digest(p) for p in sorted(args.output_dir.iterdir())},
        },
    )


if __name__ == "__main__":
    main()
