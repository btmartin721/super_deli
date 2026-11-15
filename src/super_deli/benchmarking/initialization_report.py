"""Small descriptive summaries with independent parents and conditional draws."""

from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import beta

from super_deli.benchmarking.full_retention_report import _stability
from super_deli.benchmarking.initialization_config import ARMS
from super_deli.benchmarking.signal_loss_report import interval
from super_deli.io.config import read_json


def _parent(row, common):
    directory = Path(row["path"])
    events = read_json(directory / "observed-pipeline/events.json")
    reps = read_json(directory / "bootstrap_replicates.json")
    base = {frozenset(i for i, label in enumerate(common) if label == j) for j in set(common)}
    recovered = 0
    unique = []
    eligibility = Counter()
    batches = Counter()
    for event in events:
        for candidate in event.get("candidates", []):
            info = candidate.get("initialization")
            if info:
                batches[str(info["selected_batch"])] += 1
    for event, rep in zip(events[1:], reps, strict=True):
        unique.append(len(set(event["draw"])))
        k4 = next((r for r in event.get("candidates", []) if r["k"] == 4), None)
        eligibility[k4["status"] if k4 else "withheld_before_search"] += 1
        if rep["status"] == "complete":
            labels = rep["labels"]
            partition = {
                frozenset(i for i, label in enumerate(labels) if label == j) for j in set(labels)
            }
            recovered += sum(group in partition for group in base) / len(base)
    requested = row["configuration"]["n_bootstrap"]
    return dict(
        common_point_cluster_recovery_lower=recovered / requested if requested else None,
        bootstrap_unique_original_range=[min(unique), max(unique)] if unique else None,
        bootstrap_k4_status=dict(eligibility),
        chosen_batch_counts=dict(batches),
    )


def aggregate(summary):
    fresh = [r for r in summary["records"] if r["origin"] == "fresh"]
    cells = []
    paired = []
    for cell in range(10):
        chosen = [r for r in fresh if r["task"]["cell"] == cell]
        if not chosen:
            continue
        task = chosen[0]["task"]
        field = "control_split" if task["law"] == "continuous_control" else "correct_recovery"
        descriptive = task["law"] == "conflicting"
        parent_metrics = {}
        for arm in ARMS:
            rows = [r["arms"][arm] for r in chosen]
            complete = [r for r in rows if r["status"] == "complete"]
            count = sum(bool(r[field]) for r in complete)
            failed = len(rows) - len(complete)
            stability = [_stability(r) for r in complete if r["configuration"]["n_bootstrap"]]
            parents = []
            for record in chosen:
                row = record["arms"][arm]
                common = record["arms"]["legacy10"]
                if (
                    row["status"] == common["status"] == "complete"
                    and row["configuration"]["n_bootstrap"]
                ):
                    parents.append(_parent(row, common["labels"]))
            parent_metrics[arm] = parents
            status = sum((Counter(p["bootstrap_k4_status"]) for p in parents), Counter())
            point_status = Counter()
            for r in complete:
                e = read_json(Path(r["path"]) / "observed-pipeline/events.json")[0]
                c = next((c for c in e["candidates"] if c["k"] == 4), None)
                point_status[c["status"] if c else "unavailable"] += 1
            cells.append(
                dict(
                    n=task["n"],
                    law=task["law"],
                    arm=arm,
                    datasets=len(rows),
                    completed=len(complete),
                    withheld=failed,
                    event="descriptive_partition_only" if descriptive else field,
                    event_count=None if descriptive else count,
                    event_interval=interval(count, len(rows))
                    if not failed and not descriptive
                    else None,
                    event_failure_bounds=None
                    if descriptive
                    else [count / len(rows), (count + failed) / len(rows)],
                    correct_k_count=sum(r["selected_k"] == r["target_k"] for r in complete)
                    if not descriptive and field != "control_split"
                    else None,
                    selected_k_counts=dict(Counter(r["selected_k"] for r in complete)),
                    median_ari=float(np.median([r["ari"] for r in complete]))
                    if complete and complete[0]["ari"] is not None
                    else None,
                    stability_parents=len(stability),
                    requested_draws=sum(r["configuration"]["n_bootstrap"] for r in rows),
                    attempted_draws=sum(s["requested"] for s in stability),
                    failed_draws=sum(s["failed"] for s in stability),
                    bootstrap_selected_k_counts=dict(
                        sum((Counter(s["selected_k_counts"]) for s in stability), Counter())
                    ),
                    median_own_point_recovery_lower=float(
                        np.median([s["mean_point_cluster_recovery_lower"] for s in stability])
                    )
                    if stability
                    else None,
                    median_common_point_recovery_lower=float(
                        np.median([p["common_point_cluster_recovery_lower"] for p in parents])
                    )
                    if parents
                    else None,
                    median_oob_pair_coverage=float(
                        np.median([s["oob_requested_pair_coverage"] for s in stability])
                    )
                    if stability
                    else None,
                    bootstrap_unique_original_range=[
                        min(p["bootstrap_unique_original_range"][0] for p in parents),
                        max(p["bootstrap_unique_original_range"][1] for p in parents),
                    ]
                    if parents
                    else None,
                    bootstrap_k4_status=dict(status),
                    point_k4_status=dict(point_status),
                    chosen_batch_counts=dict(
                        sum((Counter(p["chosen_batch_counts"]) for p in parents), Counter())
                    ),
                )
            )
        for left, right in combinations(ARMS, 2):
            outcomes = Counter()
            for record in chosen:
                a, b = record["arms"][left], record["arms"][right]
                outcomes[
                    "withheld"
                    if a["status"] != "complete" or b["status"] != "complete"
                    else "descriptive_only"
                    if descriptive
                    else f"left_{bool(a[field])}_right_{bool(b[field])}"
                ] += 1
            n = len(chosen)
            improve = outcomes["left_False_right_True"]
            worsen = outcomes["left_True_right_False"]
            # Conservative paired difference interval via simultaneous bounds for
            # the two discordance probabilities (Bonferroni,95% per contrast).
            low = [
                float(beta.ppf(0.0125, improve, n - improve + 1)) if improve else 0.0,
                float(beta.ppf(0.9875, improve + 1, n - improve)) if improve < n else 1.0,
            ]
            high = [
                float(beta.ppf(0.0125, worsen, n - worsen + 1)) if worsen else 0.0,
                float(beta.ppf(0.9875, worsen + 1, n - worsen)) if worsen < n else 1.0,
            ]
            paired.append(
                dict(
                    n=task["n"],
                    law=task["law"],
                    event="descriptive_partition_only" if descriptive else field,
                    left=left,
                    right=right,
                    datasets=n,
                    outcomes=dict(outcomes),
                    difference=(improve - worsen) / n
                    if not outcomes["withheld"] and not descriptive
                    else None,
                    difference_interval=[low[0] - high[1], low[1] - high[0]]
                    if not outcomes["withheld"] and not descriptive
                    else None,
                )
            )
    return dict(
        schema_version=1,
        phase=summary["phase"],
        fresh_datasets=len(fresh),
        cells=cells,
        paired_comparisons=paired,
        candidate_searches=summary["candidate_searches"],
        independent=summary["independent"],
        exact_parity_arrays=summary["exact_parity_arrays"],
        individual_records_published=0,
        new_p_values=0,
        new_defaults=0,
        uncertainty=(
            "Unadjusted exact95% per-cell intervals; conservative paired difference intervals "
            "per contrast; not simultaneous across cells"
        ),
        bootstrap_interpretation=(
            "Three independent parents/cell;25 draws/parent are repeated conditional measurements, "
            "not false-positive trials or species probabilities"
        ),
        limits=[
            "No empirical fitting or significance calibration.",
            "Continuous-control raw simulation is not an exact Gaussian score null.",
            "Full retention preserves selected view PCs,not every raw feature.",
            "No optimizer/default promotion from recovery alone.",
            "Conflicting views have no joint target or recovery rate.",
        ],
    )
