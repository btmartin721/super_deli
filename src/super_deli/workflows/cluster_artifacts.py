"""Shared candidate-cluster evidence and numerical artifacts for unit bootstraps."""

from __future__ import annotations

import csv

from super_deli.core.results import Assignment, DelimitationResult, Evidence
from super_deli.io.config import write_json


def candidate_result(
    dataset, fit, digest, level, *, model_name, space, pipeline, result_type=DelimitationResult
):
    point, stability = fit.point, fit.stability
    assignments, evidence = [], []
    for i, unit in enumerate(fit.unit_ids):
        label = f"cluster_{point.labels[i] + 1}"
        probability = float(point.membership[i].max())
        resolved = probability >= fit.config.assignment_threshold
        assignments.append(
            Assignment(
                unit,
                label if resolved else None,
                None if resolved else "Below the declared conditional-membership threshold",
            )
        )
        evidence.append(
            Evidence(
                "membership_probability",
                f"{unit}:{label}",
                "available",
                method=(
                    f"Selected {point.mixture.covariance_type} covariance GMM "
                    f"conditional membership in {space} space"
                ),
                value=probability,
            )
        )
    for k in range(point.selected_k):
        available = stability.status == "available"
        evidence.append(
            Evidence(
                "bootstrap_frequency",
                f"cluster_{k + 1}",
                stability.status,
                method=f"Exact original-unit-set recovery after whole-unit {pipeline} refits",
                value=stability.exact_recovery_counts[k] / stability.requested
                if available
                else None,
                replicates=stability.requested or None,
                reason=None
                if available
                else (
                    "Bootstrap disabled"
                    if not stability.requested
                    else "Requested replicate failed; see identification bounds and replicate log"
                ),
            )
        )
    evidence.append(
        Evidence(
            "p_value",
            "species_boundaries",
            "not_computed",
            reason=f"No species-boundary null model is defined by {space}/GMM clustering",
        )
    )
    return result_type(
        dataset.dataset_id,
        model_name,
        level,
        tuple(assignments),
        tuple(evidence),
        digest,
    )


def write_cluster_artifacts(fit, result, *, scores, observation_ids, directory):
    import numpy as np

    destination = directory
    point, stability = fit.point, fit.stability
    order = list(point.component_order)

    # Store one dense covariance per component for a uniform reconstruction contract.
    # Tied matrices are repeated explicitly; parameter counts still use one matrix.
    def component_matrices(values):
        if point.mixture.covariance_type == "tied":
            return np.repeat(values[None, :, :], point.selected_k, axis=0)
        return values[order]

    with (destination / "mixture.npz").open("xb") as handle:
        np.savez(
            handle,
            labels=point.labels,
            membership=point.membership,
            scores=scores,
            weights=point.mixture.weights_[order],
            means=point.mixture.means_[order],
            covariances=component_matrices(point.mixture.covariances_),
            precisions_cholesky=component_matrices(point.mixture.precisions_cholesky_),
        )
    if stability.requested:
        with (destination / "stability.npz").open("xb") as handle:
            np.savez(handle, **stability.arrays)
    with (destination / "assignments.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "observation_id",
                "unit_id",
                "candidate_cluster",
                "assignment",
                "reason",
                *(f"membership_cluster_{k + 1}" for k in range(point.selected_k)),
            )
        )
        for i, (observation, assignment) in enumerate(
            zip(observation_ids, result.assignments, strict=True)
        ):
            writer.writerow(
                (
                    observation,
                    fit.unit_ids[i],
                    f"cluster_{point.labels[i] + 1}",
                    assignment.label,
                    assignment.reason,
                    *point.membership[i],
                )
            )
    write_json(result.to_dict(), destination / "result.json")
    write_json(stability.replicates, destination / "bootstrap_replicates.json")
    write_json(point.candidates, destination / "candidates.json")
    clusters = []
    for k in range(point.selected_k):
        count = stability.exact_recovery_counts[k] if stability.requested else None
        clusters.append(
            {
                "label": f"cluster_{k + 1}",
                "unit_ids": [u for u, c in zip(fit.unit_ids, point.labels, strict=True) if c == k],
                "exact_recovery_count": count,
                "frequency": count / stability.requested
                if stability.status == "available"
                else None,
                "failure_lower": count / stability.requested if stability.requested else None,
                "failure_upper": (count + len(stability.failures)) / stability.requested
                if stability.requested
                else None,
            }
        )
    write_json(clusters, destination / "clusters.json")
