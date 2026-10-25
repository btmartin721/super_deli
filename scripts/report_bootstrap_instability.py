"""Compact descriptive reports; datasets remain the independent simulation units."""

import argparse
import csv
import hashlib
import json
import statistics
import tarfile
from collections import Counter, defaultdict
from pathlib import Path


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def precision(successes, available, requested):
    if not available:
        return dict(
            available=0,
            requested=requested,
            successes=0,
            conditional_rate=None,
            conditional_exact_interval=None,
            unknown_outcome_bounds=[0, 1],
        )
    from scipy.stats import beta

    return dict(
        available=available,
        requested=requested,
        successes=successes,
        conditional_rate=successes / available,
        conditional_exact_interval=[
            0 if successes == 0 else float(beta.ppf(0.025, successes, available - successes + 1)),
            1
            if successes == available
            else float(beta.ppf(0.975, successes + 1, available - successes)),
        ],
        unknown_outcome_bounds=[
            successes / requested,
            (successes + requested - available) / requested,
        ],
    )


def trace(records):
    full = [r["searches"].get("full", {}) for r in records]
    stages = ("raw_joint", "fused", "joint", "projected_all", "projected_global_oob", "count")
    axes = {
        stage: {
            axis: mean([(v.get(stage + "_axis_r2") or {}).get(axis) for v in full])
            for axis in ("a", "b")
        }
        for stage in stages
    }
    views = {}
    for kind in ("morphological", "ecological"):
        values = [v for r in full for v in r.get("views", []) if v["kind"] == kind]
        views[kind] = {
            stage: {
                axis: mean([(v.get(stage + "_axis_r2") or {}).get(axis) for v in values])
                for axis in ("a", "b")
            }
            for stage in ("raw_mean", "standardized", "view_pca", "source_projected", "profile")
        }
    candidates = [
        c
        for r in records
        for search in r["searches"].values()
        for c in search.get("candidates", [])
        if c["k"] == 4 and c["status"] != "insufficient_samples"
    ]
    selected = [
        next(
            (
                v
                for v in r["searches"].get("full", {}).get("candidates", [])
                if v["k"] == r["selection"].get("screened_k")
            ),
            {},
        )
        for r in records
    ]
    return dict(
        selected_all_original_ari=mean([v.get("all_original_ari") for v in selected]),
        selected_global_oob_ari=mean([v.get("global_oob_ari") for v in selected]),
        selected_training_occurrence_ari=mean([v.get("training_occurrence_ari") for v in selected]),
        global_oob_counts=[len(r["global_oob"]) for r in records],
        full_search_axis_r2=axes,
        full_view_axis_r2=views,
        k4_mean_entropy=mean([c["entropy"] for c in candidates]),
        k4_max_covariance_condition=max(
            (max(c["covariance_condition_numbers"]) for c in candidates), default=None
        ),
        k4_max_iterations=max((c["iterations"] for c in candidates), default=None),
        k4_unconverged=sum(not c["converged"] for c in candidates),
        k4_candidate_denominator=len(candidates),
    )


def summarize(records, requested, reference_k=None):
    available = [r for r in records if r.get("gate", {}).get("status") == "available"]
    k4 = [next(v for v in r["gate"]["candidates"] if v["k"] == 4) for r in available]
    failures = Counter(r["first_failing_gate"] for r in k4)
    selected4 = sum(r["selection"]["screened_k"] == 4 for r in available)
    same = [r for r in available if r["selection"]["screened_k"] == reference_k]
    return dict(
        reference_point_k=reference_k,
        same_point_k_draws=len(same),
        mean_assignment_changes_when_same_k=mean([r["point_assignment_changes"] for r in same]),
        signal_and_numerical_trace=trace(available),
        requested=requested,
        available=len(available),
        selected_k_counts=dict(Counter(str(r["selection"]["screened_k"]) for r in available)),
        k4=precision(selected4, len(available), requested),
        k_gt1=precision(
            sum(r["selection"]["screened_k"] > 1 for r in available), len(available), requested
        ),
        k4_first_gate=dict(failures),
        k4_overlapping_failures=dict(
            total_feasibility=sum(
                not (v["full_distinct_feasible"] and v["all_fold_distinct_feasible"]) for v in k4
            ),
            full_eligibility=sum(not v["full_eligible"] for v in k4),
            fold_eligibility=sum(not v["all_folds_eligible"] for v in k4),
            nonpositive_gain=sum(
                v["pooled_validation_gain"] is not None and v["pooled_validation_gain"] <= 0
                for v in k4
            ),
        ),
        generating_partition_full_feasible=sum(
            v["generating_full_sizes"] is not None and min(v["generating_full_sizes"]) >= 5
            for v in k4
        ),
        generating_partition_all_folds_feasible=sum(
            all(v is not None and min(v) >= 5 for v in r["generating_fold_sizes"]) for r in k4
        ),
        mean_selected_ari=mean([r["selected_ari"] for r in available]),
        mean_assignment_changes=mean([r["point_assignment_changes"] for r in available]),
        mean_distinct=mean([len(set(r["draw_indices"])) for r in available]),
        mean_k4_validation_gain=mean([r["pooled_validation_gain"] for r in k4]),
        mean_k4_equal_unit_gain=mean([r["equal_original_validation_gain"] for r in k4]),
        counterfactual_available_counts={
            policy: sum(
                r["gate"]["counterfactuals"][policy]["status"] == "available" for r in available
            )
            for policy in (
                "remove_full_size",
                "remove_fold_size",
                "remove_gain",
                "occurrence_sizes",
                "equal_validation_weights",
            )
        },
        counterfactual_k4_counts={
            policy: sum(
                r["gate"]["counterfactuals"][policy].get("selected_k") == 4 for r in available
            )
            for policy in (
                "remove_full_size",
                "remove_fold_size",
                "remove_gain",
                "occurrence_sizes",
                "equal_validation_weights",
            )
        },
        mean_joint_axis_a=mean(
            [
                r["searches"]["full"].get("joint_axis_r2", {}).get("a")
                if r["searches"]["full"].get("joint_axis_r2")
                else None
                for r in available
            ]
        ),
        mean_joint_axis_b=mean(
            [
                r["searches"]["full"].get("joint_axis_r2", {}).get("b")
                if r["searches"]["full"].get("joint_axis_r2")
                else None
                for r in available
            ]
        ),
    )


def cohort(source, archived=False):
    source = Path(source)
    rows = [json.loads(s) for s in (source / "observations.jsonl").read_text().splitlines()]
    result = []
    oracle_rows = []
    for row in sorted(rows, key=lambda r: (r["cell"], r["repetition"])):
        with tarfile.open(source / row["archive"], "r|gz") as archive:
            # Read the compressed stream once. Random backward seeks would
            # repeatedly decompress raw fitted arrays that reporting never uses.
            files = {
                member.name: json.load(archive.extractfile(member))
                for member in archive
                if member.isfile()
                and member.name.endswith(".json")
                and (
                    archived
                    or member.name.startswith("diagnostics/")
                    or member.name in {"workflows.json", "oracles.json"}
                )
            }

            def load(name, files=files):
                return files[name]

            if archived:
                arms = {
                    "baseline": dict(
                        point=load("point.json"),
                        draws=[load(f"draw_{i:04d}.json") for i in range(99)],
                    )
                }
            else:
                workflows = load("workflows.json")
                arms = defaultdict(lambda: dict(point=None, draws=[]))
                for item in workflows:
                    arm, i = item["arm"], item["replicate"]
                    name = "point" if i == -1 else f"draw_{i:04d}"
                    value = load(f"diagnostics/{arm}/{name}.json")
                    if i == -1:
                        arms[arm]["point"] = value
                    else:
                        arms[arm]["draws"].append(value)
                for oracle in load("oracles.json"):
                    reference = load(f"diagnostics/baseline/{oracle['workflow']}.json")
                    candidate = next(
                        (
                            v
                            for v in reference["searches"][oracle["prefix"]].get("candidates", [])
                            if v["k"] == 4
                        ),
                        None,
                    )
                    oracle_rows.append(
                        dict(
                            cell=row["cell"],
                            repetition=row["repetition"],
                            workflow=oracle["workflow"],
                            prefix=oracle["prefix"],
                            status=oracle["status"],
                            eligible=oracle.get("eligible_original_ids"),
                            likelihood_gain_over_ordinary_k4=oracle["log_likelihood"]
                            - candidate["log_likelihood"]
                            if oracle["status"] == "complete"
                            and candidate
                            and "log_likelihood" in candidate
                            else None,
                            legacy_occurrence_eligibility=oracle.get(
                                "legacy_occurrence_eligibility"
                            ),
                        )
                    )
            for arm, records in arms.items():
                point = records["point"]
                identity = dict(
                    cohort="archived" if archived else "fresh",
                    cell=row["cell"],
                    repetition=row["repetition"],
                    arm=arm,
                    point_k=None if point is None else point.get("selection", {}).get("screened_k"),
                    point_ari=None if point is None else point.get("selected_ari"),
                )
                for prefix, values in (
                    ("all", records["draws"]),
                    ("first_five", records["draws"][:5]),
                ):
                    result.append(
                        dict(
                            **identity,
                            draw_subset=prefix,
                            **summarize(
                                values,
                                len(values),
                                arms["baseline"]["point"]["selection"].get("screened_k"),
                            ),
                        )
                    )
    return result, oracle_rows


def report(archived, study, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    old, _ = cohort(archived, True)
    fresh, oracles = cohort(study)
    datasets = old + fresh
    groups = defaultdict(list)
    for row in datasets:
        groups[row["cohort"], row["cell"], row["arm"], row["draw_subset"]].append(row)
    cells = []
    for (cohort_name, cell, arm, subset), rows in sorted(groups.items()):
        frequencies = [
            r["k4"]["conditional_rate"] for r in rows if r["k4"]["conditional_rate"] is not None
        ]
        cells.append(
            dict(
                cohort=cohort_name,
                cell=cell,
                arm=arm,
                draw_subset=subset,
                datasets=len(rows),
                point_k4=sum(r["point_k"] == 4 for r in rows),
                point_k_gt1=sum(r["point_k"] is not None and r["point_k"] > 1 for r in rows),
                mean_dataset_k_gt1=mean([r["k_gt1"]["conditional_rate"] for r in rows]),
                mean_dataset_k4=mean(frequencies),
                dataset_k4_range=[min(frequencies), max(frequencies)] if frequencies else None,
                mean_dataset_ari=mean([r["mean_selected_ari"] for r in rows]),
                mean_assignment_changes=mean([r["mean_assignment_changes"] for r in rows]),
                mean_distinct=mean([r["mean_distinct"] for r in rows]),
                mean_axis_a=mean([r["mean_joint_axis_a"] for r in rows]),
                mean_axis_b=mean([r["mean_joint_axis_b"] for r in rows]),
                first_gate_counts={
                    k: sum(r["k4_first_gate"].get(k, 0) for r in rows)
                    for k in (
                        "distinct_feasibility",
                        "full_eligibility",
                        "fold_eligibility",
                        "nonpositive_gain",
                        "aic_ranking",
                        "selected",
                    )
                },
                overlap_counts={
                    k: sum(r["k4_overlapping_failures"][k] for r in rows)
                    for k in (
                        "total_feasibility",
                        "full_eligibility",
                        "fold_eligibility",
                        "nonpositive_gain",
                    )
                },
                generating_full_feasible=sum(r["generating_partition_full_feasible"] for r in rows),
                generating_all_folds_feasible=sum(
                    r["generating_partition_all_folds_feasible"] for r in rows
                ),
                counterfactual_k4_counts={
                    k: sum(r["counterfactual_k4_counts"][k] for r in rows)
                    for k in rows[0]["counterfactual_k4_counts"]
                },
            )
        )
    paired = []
    lookup = {
        (r["cell"], r["repetition"], r["arm"]): r for r in fresh if r["draw_subset"] == "first_five"
    }
    for (cell, rep, arm), base in sorted(lookup.items()):
        if arm != "baseline":
            continue
        for other in ("unique", "extended"):
            contrast = lookup[cell, rep, other]
            paired.append(
                dict(
                    cell=cell,
                    repetition=rep,
                    comparison=other + "_minus_baseline",
                    k4_rate_change=contrast["k4"]["conditional_rate"]
                    - base["k4"]["conditional_rate"]
                    if contrast["k4"]["conditional_rate"] is not None
                    and base["k4"]["conditional_rate"] is not None
                    else None,
                    ari_change=contrast["mean_selected_ari"] - base["mean_selected_ari"]
                    if contrast["mean_selected_ari"] is not None
                    and base["mean_selected_ari"] is not None
                    else None,
                )
            )
    results = dict(
        datasets=datasets,
        cells=cells,
        paired=paired,
        oracle_diagnostics=oracles,
        interpretation=(
            "Original datasets are independent; draws/folds repeat measurements. "
            "Conditional binomial intervals are unadjusted Monte Carlo precision only. "
            "Paired ten-dataset effects are descriptive. Unique support changes weights/sample "
            "size/maps/order. Oracle fits never enter selection. "
            "No new p-values or species guarantees."
        ),
    )
    (output / "summary.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    for name, rows in (("cells", cells), ("paired", paired)):
        keys = list(rows[0])
        with (output / (name + ".csv")).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                        for k, v in row.items()
                    }
                )
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()
    }
    (output / "checksums.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    return dict(
        status="complete",
        dataset_summaries=len(datasets),
        cell_summaries=len(cells),
        paired=len(paired),
        oracle_diagnostics=len(oracles),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archived", type=Path, required=True)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(report(args.archived, args.study, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
