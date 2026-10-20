"""Ingest landmarks and export auditable candidate shape clusters and stability."""

from __future__ import annotations

import csv
from importlib.metadata import version
from pathlib import Path

from super_deli.core.data import DatasetConfig
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.results import Assignment, DelimitationResult, Evidence
from super_deli.io.config import read_json, write_json
from super_deli.models.shape_cluster_config import ShapeClusterConfig
from super_deli.workflows.shape_pca import run_shape_pca


def _result(dataset, fit, digest, level):
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
                method="Selected full covariance GMM conditional membership in shape PCA space",
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
                method="Exact original-unit-set recovery after whole-unit GPA/PCA/GMM refits",
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
            reason="No species-boundary null model is defined by shape PCA/GMM clustering",
        )
    )
    return DelimitationResult(
        dataset.dataset_id,
        "shape_pca_full_gmm",
        level,
        tuple(assignments),
        tuple(evidence),
        digest,
    )


def run_shape_gmm(
    dataset: DatasetConfig,
    *,
    model_config: ShapeClusterConfig,
    base_dir: str | Path,
    output_dir: str | Path,
    modality_name: str | None = None,
) -> dict:
    """Reuse shape ingestion/exports; write success only after the clustering artifacts."""
    model_config = ShapeClusterConfig.from_dict(model_config.to_dict())
    try:
        import numpy as np
        from threadpoolctl import threadpool_limits

        from super_deli.models.mixture import FIT_LOCK
        from super_deli.models.shape_gmm import _cluster_fitted_shape
        from super_deli.models.shape_pca import ShapePCAFit
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Shape clustering requires: pip install 'super-deli[clustering]'"
        ) from exc
    destination = Path(output_dir).absolute()
    destination.mkdir(exist_ok=False)
    try:
        with FIT_LOCK, threadpool_limits(limits=model_config.n_threads):
            shape = run_shape_pca(
                dataset,
                model_config=model_config.shape_config,
                base_dir=base_dir,
                output_dir=destination / "shape",
                modality_name=modality_name,
            )
            records = shape.retained.records
            fit = _cluster_fitted_shape(
                [r.coordinates for r in records],
                [r.unit_id for r in records],
                config=model_config,
                shape=ShapePCAFit(
                    shape.model,
                    shape.projection,
                    tuple(read_json(shape.output_dir / "gpa_history.json")),
                ),
            )
        configuration = {
            "dataset": dataset.to_dict(),
            "model": model_config.to_dict(),
            "modality": shape.retained.modality_name,
        }
        digest = configuration_hash(configuration)
        result = _result(dataset, fit, digest, shape.retained.level)
        point, stability = fit.point, fit.stability
        order = list(point.component_order)
        with (destination / "mixture.npz").open("xb") as handle:
            np.savez(
                handle,
                labels=point.labels,
                membership=point.membership,
                scores=fit.shape.projection.scores,
                weights=point.mixture.weights_[order],
                means=point.mixture.means_[order],
                covariances=point.mixture.covariances_[order],
                precisions_cholesky=point.mixture.precisions_cholesky_[order],
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
            for i, (record, assignment) in enumerate(zip(records, result.assignments, strict=True)):
                writer.writerow(
                    (
                        record.observation_id,
                        record.unit_id,
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
                    "unit_ids": [
                        u for u, c in zip(fit.unit_ids, point.labels, strict=True) if c == k
                    ],
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
        summary = {
            "schema_version": 1,
            "status": "complete",
            "model": result.model,
            "dataset_id": dataset.dataset_id,
            "modality": shape.retained.modality_name,
            "level": shape.retained.level,
            "observation_ids": [r.observation_id for r in records],
            "unit_ids": list(fit.unit_ids),
            "excluded_observations": shape.summary["excluded_observations"],
            "selected_k": point.selected_k,
            "covariance_type": "full",
            "selection": "Minimum BIC among size-admissible K=1..max_k on a fixed PC dimension",
            "n_pcs": model_config.n_pcs,
            "point_seed": fit.point_seed,
            "bootstrap": {
                "status": stability.status,
                "requested": stability.requested,
                "successful": stability.successful,
                "failed": len(stability.failures),
                "selected_k_counts": stability.selected_k_counts,
                "resampling_unit": "independent biological unit; all landmarks retained together",
                "pipeline": "Refit GPA, tangent PCA and BIC/GMM on each bootstrap draw",
                "evaluation": "Project all original units into each replicate's fitted shape space",
                "denominator": "All requested draws; failed outcomes bounded without retries",
                "oob": "Pair-specific counts from draws excluding both units; inspect denominators",
                "intervals": "Failure identification bounds, not confidence intervals",
            },
            "interpretation": [
                "Mixture components are candidate shape groups, not delimited species",
                "Conditional membership is not bootstrap frequency or species probability",
                "Original-unit stability includes in-bag observations and can be optimistic",
                "Out-of-bag diagnostics may have small or zero pairwise denominators",
                "Bootstrap stability is conditional on sampling, landmark and model assumptions",
                "No species null test, confidence intervals, landmark error or allometry model",
            ],
            "provenance": {
                "configuration": configuration,
                "configuration_sha256": digest,
                "shape": shape.summary["provenance"],
                "mixture_array_sha256": fingerprint_file(destination / "mixture.npz").sha256,
                "software": {
                    name: version(name)
                    for name in ("super-deli", "numpy", "scipy", "scikit-learn", "threadpoolctl")
                },
            },
        }
        write_json(summary, destination / "summary.json")
        return summary
    except Exception as exc:
        write_json({"status": "failed", "error": str(exc)}, destination / "failure.json")
        raise
