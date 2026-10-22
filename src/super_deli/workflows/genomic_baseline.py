"""Run the genomic baseline without using population labels as predictors."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import numpy as np

from super_deli.core.data import DatasetConfig
from super_deli.core.provenance import configuration_hash
from super_deli.core.results import Assignment, DelimitationResult, Evidence
from super_deli.io.config import write_json
from super_deli.io.genomic import _select_modality, ingest_genomic
from super_deli.io.genomic_encoding import encode_diploid
from super_deli.models.block_map import parse_blocks as parse_blocks
from super_deli.models.config import BaselineConfig
from super_deli.models.pca_gmm import BaselineFit, fit_pca_gmm


def _result(dataset_id: str, fit: BaselineFit, digest: str) -> DelimitationResult:
    point, stability = fit.point, fit.stability
    assignments, evidence = [], []
    for i, unit_id in enumerate(fit.specimen_ids):
        label = f"cluster_{point.labels[i] + 1}"
        probability = float(point.membership[i].max())
        resolved = probability >= fit.config.assignment_threshold
        assignments.append(
            Assignment(
                unit_id,
                label if resolved else None,
                None if resolved else "Below the declared conditional-membership threshold",
            )
        )
        evidence.append(
            Evidence(
                "membership_probability",
                f"{unit_id}:{label}",
                "available",
                method="Selected diagonal GMM conditional membership in the fitted PCA space",
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
                method="Exact specimen-set recovery after whole-block pipeline refits",
                value=stability.exact_recovery_counts[k] / stability.requested
                if available
                else None,
                replicates=stability.requested or None,
                reason=None
                if available
                else "Bootstrap disabled"
                if not stability.requested
                else "A requested replicate failed; see failure bounds and replicate records",
            )
        )
    evidence.append(
        Evidence(
            "p_value",
            "species_boundaries",
            "not_computed",
            reason="No species-boundary null model is defined by PCA/GMM clustering",
        )
    )
    return DelimitationResult(
        dataset_id,
        "genomic_pca_diagonal_gmm",
        "specimen",
        tuple(assignments),
        tuple(evidence),
        digest,
    )


def run_genomic_baseline(
    dataset: DatasetConfig,
    *,
    model_config: BaselineConfig,
    base_dir: str | Path,
    output_dir: str | Path,
    block_ids: tuple[str, ...] | None = None,
    modality_name: str | None = None,
) -> dict:
    """Ingest one source, fit candidate clusters, and write a new artifact directory.

    A completed analysis can have unavailable bootstrap evidence; point fitting
    failures create failure.json instead of a completion summary. No source file
    or existing output directory is overwritten.
    """
    modality = _select_modality(dataset, modality_name)
    ids = [o.unit_id for o in modality.observations]
    if len(ids) != len(set(ids)):
        raise ValueError(
            "Repeated specimens require an aggregation policy before baseline analysis"
        )
    destination = Path(output_dir).absolute()
    destination.mkdir(exist_ok=False)
    try:
        loaded = ingest_genomic(
            dataset,
            base_dir=base_dir,
            output_dir=destination / "ingestion",
            modality_name=modality.name,
        )
        dosages, encoding = encode_diploid(
            loaded, output_prefix=destination / "encoding", random_seed=model_config.random_seed
        )
        fit = fit_pca_gmm(dosages, loaded.specimen_ids, config=model_config, block_ids=block_ids)
        configuration = {
            "dataset": dataset.to_dict(),
            "model": model_config.to_dict(),
            "modality": modality.name,
            "blocks": {"schema_version": 1, "block_ids": block_ids}
            if block_ids is not None
            else None,
        }
        digest = configuration_hash(configuration)
        result = _result(dataset.dataset_id, fit, digest)
        point, stability = fit.point, fit.stability
        for name, array in {
            "dosages": dosages,
            "labels": point.labels,
            "membership": point.membership,
            "pca_scores": point.scores,
        }.items():
            np.save(destination / f"{name}.npy", array, allow_pickle=False)
        pre = point.preprocessing
        np.savez(
            destination / "preprocessing.npz",
            retained_locus_indices=pre.kept_indices,
            means=pre.means,
            scales=pre.scales,
            pca_components=point.pca.components_,
            pca_mean=point.pca.mean_,
            explained_variance_ratio=point.explained_variance_ratio,
            sample_missing_fraction=pre.sample_missing_fraction,
        )
        # Parameters are exported in the same canonical component order as membership.
        order = list(point.component_order)
        np.savez(
            destination / "mixture.npz",
            weights=point.mixture.weights_[order],
            means=point.mixture.means_[order],
            variances=point.mixture.covariances_[order],
        )
        for name, array in {
            "coassignment": stability.coassignment,
            "coassignment_lower": stability.coassignment_lower,
            "coassignment_upper": stability.coassignment_upper,
        }.items():
            if array is not None:
                np.save(destination / f"{name}.npy", array, allow_pickle=False)
        write_json(result.to_dict(), destination / "result.json")
        write_json(stability.replicates, destination / "bootstrap_replicates.json")
        summary = {
            "schema_version": 1,
            "status": "complete",
            "inference_scope": "candidate_genetic_clusters",
            "dataset_id": dataset.dataset_id,
            "modality": modality.name,
            "sample_ids": loaded.sample_ids,
            "specimen_ids": loaded.specimen_ids,
            "cluster_labels": [f"cluster_{k + 1}" for k in range(point.selected_k)],
            "selected_k": point.selected_k,
            "selected_k_at_search_limit": point.selected_k == model_config.max_k,
            "candidates": point.candidates,
            "encoding": encoding,
            "preprocessing": {
                "retained_locus_count": len(pre.kept_indices),
                "exclusions": pre.exclusions,
                "imputed_call_count": pre.imputed_calls,
                "sample_missing_fraction": pre.sample_missing_fraction.tolist(),
            },
            "stability": {
                "status": stability.status,
                "requested": stability.requested,
                "successful": stability.successful,
                "failures": stability.failures,
                "block_source": stability.block_source,
                "block_count": stability.block_count,
                "selected_k_counts_successful": stability.selected_k_counts,
                "exact_cluster_recovery_counts": stability.exact_recovery_counts,
                "exact_cluster_recovery_failure_bounds": [
                    [
                        count / stability.requested,
                        (count + len(stability.failures)) / stability.requested,
                    ]
                    for count in stability.exact_recovery_counts
                ]
                if stability.requested
                else [],
                "denominator": "all requested replicates",
                "bounds_interpretation": "Unresolved-replicate bounds; not confidence intervals",
                "pairwise_statistic": "Fraction of whole-block refits assigning a pair together",
                "cluster_statistic": "Whole-block frequency of exact specimen-set recovery",
            },
            "interpretation": [
                "Candidate genetic clusters do not establish species boundaries",
                "Membership probabilities are conditional on the selected fitted mixture",
                "Bootstrap stability is conditional on the sampled specimens",
                "Spatial structure, relatedness, ascertainment and missingness can induce clusters",
                "Mean imputation omits genotype uncertainty and does not correct missingness bias",
                "K selection depends on the declared PCA dimension and candidate search range",
                "Block independence and specimen independence are user assumptions",
            ],
            "provenance": {
                "configuration": configuration,
                "configuration_sha256": digest,
                "dataset_configuration_sha256": loaded.summary["provenance"][
                    "configuration_sha256"
                ],
                "ingestion": loaded.summary["provenance"],
                "software": {
                    name: version(name)
                    for name in (
                        "super-deli",
                        "numpy",
                        "scipy",
                        "scikit-learn",
                        "threadpoolctl",
                        "snpio",
                    )
                },
            },
        }
        write_json(summary, destination / "summary.json")
        return summary
    except Exception as exc:
        write_json({"status": "failed", "error": str(exc)}, destination / "failure.json")
        raise
