"""Aggregate-only descriptive validation; bootstrap draws are repeated measurements."""

from collections import Counter
from pathlib import Path

import numpy as np

from super_deli.benchmarking.signal_loss_report import interval
from super_deli.io.config import read_json


def _stability(row):
    root = Path(row["path"])
    replicates = read_json(root / "bootstrap_replicates.json")
    requested = len(replicates)
    successful = [r for r in replicates if r["status"] == "complete"]
    if not requested:
        return None
    clusters = read_json(root / "clusters.json")
    recovery = np.mean([r["exact_recovery_count"] / requested for r in clusters])
    counts = Counter(r["selected_k"] for r in successful)
    with np.load(root / "stability.npz", allow_pickle=False) as a:
        mask = ~np.eye(len(a["oob_requested"]), dtype=bool)
        oob_coverage = np.mean(a["oob_requested"][mask] > 0)
        conditional = np.mean(a["oob_successful"][mask] > 0)
    return dict(
        requested=requested,
        successful=len(successful),
        failed=requested - len(successful),
        selected_k_counts=dict(counts),
        mean_point_cluster_recovery_lower=float(recovery),
        mean_point_cluster_recovery_upper=float(
            recovery + (requested - len(successful)) / requested
        ),
        oob_requested_pair_coverage=float(oob_coverage),
        oob_successful_pair_coverage=float(conditional),
    )


def aggregate(summary):
    fresh = [r for r in summary["records"] if r["origin"] == "fresh"]
    cells, paired = [], []
    for cell in range(8):
        chosen = [r for r in fresh if r["task"]["cell"] == cell]
        if not chosen:
            continue
        task = chosen[0]["task"]
        for arm in ("legacy_two", "matched_two", "retained_all"):
            rows = [r["arms"][arm] for r in chosen]
            complete = [r for r in rows if r["status"] == "complete"]
            failure = len(rows) - len(complete)
            field = "control_split" if task["law"] == "continuous_control" else "correct_recovery"
            count = sum(bool(r[field]) for r in complete)
            stability = [_stability(r) for r in complete if r["stability"]["requested"]]
            cells.append(
                dict(
                    n=task["n"],
                    law=task["law"],
                    arm=arm,
                    datasets=len(rows),
                    completed=len(complete),
                    failed=failure,
                    selected_k_counts=dict(Counter(r["selected_k"] for r in complete)),
                    event=field,
                    event_count=count,
                    event_interval=interval(count, len(rows)) if failure == 0 else None,
                    correct_k_count=sum(r["selected_k"] == r["target_k"] for r in complete)
                    if task["law"] != "continuous_control"
                    else None,
                    event_failure_bounds=[count / len(rows), (count + failure) / len(rows)],
                    median_ari=float(np.median([r["ari"] for r in complete]))
                    if complete and complete[0]["ari"] is not None
                    else None,
                    stability_parents=len(stability),
                    planned_stability_parents=sum(
                        r["configuration"]["n_bootstrap"] > 0 for r in rows
                    ),
                    requested_draws=sum(r["configuration"]["n_bootstrap"] for r in rows),
                    attempted_draws=sum(r["requested"] for r in stability),
                    unattempted_draws=sum(
                        r["configuration"]["n_bootstrap"] for r in rows if r["status"] != "complete"
                    ),
                    failed_draws=sum(r["failed"] for r in stability),
                    stability_k_counts=dict(
                        sum((Counter(r["selected_k_counts"]) for r in stability), Counter())
                    ),
                    median_parent_cluster_recovery_lower=float(
                        np.median([r["mean_point_cluster_recovery_lower"] for r in stability])
                    )
                    if stability
                    else None,
                    median_parent_cluster_recovery_upper=float(
                        np.median([r["mean_point_cluster_recovery_upper"] for r in stability])
                    )
                    if stability
                    else None,
                    median_parent_oob_pair_coverage=float(
                        np.median([r["oob_requested_pair_coverage"] for r in stability])
                    )
                    if stability
                    else None,
                    median_parent_successful_oob_coverage=float(
                        np.median([r["oob_successful_pair_coverage"] for r in stability])
                    )
                    if stability
                    else None,
                )
            )
        outcomes = Counter()
        for r in chosen:
            a, b = r["arms"]["matched_two"], r["arms"]["retained_all"]
            if a["status"] != "complete" or b["status"] != "complete":
                outcomes["withheld"] += 1
            else:
                outcomes[f"two_{bool(a[field])}_all_{bool(b[field])}"] += 1
        paired.append(
            dict(
                n=task["n"],
                law=task["law"],
                event=field,
                datasets=len(chosen),
                outcomes=dict(outcomes),
            )
        )
    diagnostic = []
    for origin in ("fresh", "archived"):
        rows = [r for r in summary["diagnostics"] if r["origin"] == origin]
        counts = Counter()
        for r in rows:
            base, high, truth = r["baseline"], r["hundred_starts"], r["truth_initialized"]
            counts["baseline_correct"] += bool(base.get("correct_recovery"))
            counts["hundred_start_correct"] += bool(high.get("correct_recovery"))
            counts["hundred_start_withheld"] += high["status"] != "complete"
            counts["truth_withheld"] += truth["status"] != "complete"
            counts["truth_eligible_and_better_bic"] += bool(
                truth.get("eligible") and truth.get("beats_native_selected_bic")
            )
            if base["status"] == high["status"] == "complete":
                counts["high_only_correct"] += bool(
                    high["correct_recovery"] and not base["correct_recovery"]
                )
                counts["baseline_only_correct"] += bool(
                    base["correct_recovery"] and not high["correct_recovery"]
                )
        diagnostic.append(
            dict(origin=origin, n=64, law="shape_only", datasets=len(rows), counts=dict(counts))
        )
    return dict(
        schema_version=1,
        phase=summary["phase"],
        fresh_datasets=len(fresh),
        cells=cells,
        paired_comparisons=paired,
        optimization_diagnostics=diagnostic,
        candidate_searches=summary["candidate_searches"],
        planned_truth_fits=summary["planned_truth_fits"],
        exact_parity_arrays=summary["exact_parity_arrays"],
        independent=summary["independent"],
        individual_records_published=0,
        new_p_values=0,
        new_defaults=0,
        uncertainty=(
            "Unadjusted exact 95% intervals within independent fixed-N/law cells; "
            "withheld after any missing point outcome"
        ),
        stability_interpretation=(
            "Three independent parent datasets/cell; draws are repeated "
            "conditional refits, not biological replication or species probabilities"
        ),
        limits=[
            "Retaining view PCs does not preserve discarded raw-feature information.",
            "No new calibration or empirical species-boundary inference.",
            "No absolute BIC ranking across coordinate dimensions.",
            "Hundred-start/truth-informed diagnostics never replace native ten-start selection.",
            "No heavy-tail, spatial, missing-view or grouped-sampling generality guarantee.",
        ],
    )
