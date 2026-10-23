"""Family-explicit reporting for new grouped calibration studies.

Earlier collectors remain immutable because their source hashes are part of
published study evidence. New studies share this explicit-family collector.
"""

from collections import Counter

from report_shared_calibration import digest, read, save, table
from report_shared_grouped_calibration import NAMES


def collect(root, output, *, covariance_type):
    """Use audited relative paths and retain every outcome, including withheld tests."""
    summary = read(root / "study-audit/summary.json")
    hashes = read(root / "study-audit/files.json")
    calibration = summary["study"]["calibration"]
    if (
        calibration["schema_version"] != 2
        or calibration["model"].get("covariance_type", "full") != covariance_type
    ):
        raise ValueError("Expected the declared schema-2 covariance family")

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
