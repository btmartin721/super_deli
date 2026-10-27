"""One-table morphology/ecology clustering with explicit numerical preprocessing."""

from __future__ import annotations

import csv
from importlib.metadata import version
from pathlib import Path

from super_deli.core.data import DatasetConfig
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.io.config import write_json
from super_deli.io.tabular import ingest_tabular
from super_deli.models.tabular_cluster_config import TabularClusterConfig
from super_deli.workflows.cluster_artifacts import candidate_result, write_cluster_artifacts


def run_tabular_gmm(
    dataset: DatasetConfig,
    *,
    model_config: TabularClusterConfig,
    base_dir: str | Path,
    output_dir: str | Path,
    modality_name: str | None = None,
    preprocessing_context=None,
) -> dict:
    dataset = config_from_dict(dataset.to_dict())
    model_config = TabularClusterConfig.from_dict(model_config.to_dict())
    selected = [
        m
        for m in dataset.modalities
        if m.kind in {"morphological", "ecological"}
        and (modality_name is None or m.name == modality_name)
    ]
    if len(selected) != 1:
        raise ValueError("Select exactly one morphology/ecology table with --modality")
    try:
        import numpy as np

        from super_deli.models.tabular_gmm import fit_tabular_gmm
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Tabular clustering requires: pip install 'super-deli[clustering]'"
        ) from exc
    destination = Path(output_dir).absolute()
    destination.mkdir(exist_ok=False)
    try:
        loaded = ingest_tabular(
            dataset,
            base_dir=base_dir,
            output_dir=destination / "ingestion",
            modality_names=[selected[0].name],
        )
        table = loaded.tables[0]
        if preprocessing_context is None:
            fit = fit_tabular_gmm(table, config=model_config)
        else:
            fit = fit_tabular_gmm(table, config=model_config, context=preprocessing_context)
            preprocessing_context.save(destination)
        configuration = {
            "dataset": dataset.to_dict(),
            "model": model_config.to_dict(),
            "modality": table.modality_name,
        }
        if preprocessing_context is not None:
            configuration["preprocessing_context"] = preprocessing_context.declaration()
        digest = configuration_hash(configuration)
        result = candidate_result(
            dataset,
            fit,
            digest,
            table.level,
            model_name=f"{table.kind}_pca_full_gmm",
            space="tabular PCA",
            pipeline="preprocessing/PCA/GMM",
        )
        write_cluster_artifacts(
            fit,
            result,
            scores=fit.scores,
            observation_ids=fit.observation_ids,
            directory=destination,
        )
        model = fit.model
        with (destination / "scores.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                ("observation_id", "unit_id", *(f"PC{i + 1}" for i in range(model_config.n_pcs)))
            )
            for observation, unit, scores in zip(
                fit.observation_ids, fit.unit_ids, fit.scores, strict=True
            ):
                writer.writerow((observation, unit, *scores))
        with (destination / "preprocessing.npz").open("xb") as handle:
            np.savez(
                handle,
                kept_indices=model.kept_indices,
                means=model.means,
                scales=model.scales,
                pca_mean=model.pca.mean_,
                components=model.pca.components_,
                explained_variance=model.pca.explained_variance_,
                explained_variance_ratio=model.pca.explained_variance_ratio_,
                singular_values=model.pca.singular_values_,
                sample_missing_fraction=model.training_missing_fraction,
            )
        preprocessing = {
            "schema_version": 1,
            "feature_order": model.feature_names,
            "feature_types": model.feature_types,
            "retained_features": [model.feature_names[i] for i in model.kept_indices],
            "excluded_features": model.exclusions,
            "nonselected_columns": [
                f.name for f in table.features if f.name not in model.feature_names
            ],
            "missing_policy": model_config.missing_policy,
            "imputed_cells": model.imputed_cells,
            "count_transform": model_config.count_transform,
            "scaling": model_config.scaling,
            "standard_variance_denominator": "N after explicit mean imputation (ddof=0)",
            "pca": "Full SVD, centered covariance PCA without whitening; variance ddof=1",
            "explained_variance_ratio": model.pca.explained_variance_ratio_.tolist(),
            "array_sha256": fingerprint_file(destination / "preprocessing.npz").sha256,
        }
        write_json(preprocessing, destination / "preprocessing.json")
        stability = fit.stability
        summary = {
            "schema_version": 1,
            "status": "complete",
            "model": result.model,
            "dataset_id": dataset.dataset_id,
            "modality": table.modality_name,
            "kind": table.kind,
            "level": table.level,
            "observation_ids": list(fit.observation_ids),
            "unit_ids": list(fit.unit_ids),
            "selected_k": fit.point.selected_k,
            "n_pcs": model_config.n_pcs,
            "covariance_type": "full",
            "point_seed": fit.point_seed,
            "selection": "Minimum BIC among size-admissible K=1..max_k within a fixed PC dimension",
            "preprocessing": preprocessing,
            "bootstrap": {
                "status": stability.status,
                "requested": stability.requested,
                "successful": stability.successful,
                "failed": len(stability.failures),
                "selected_k_counts": stability.selected_k_counts,
                "resampling_unit": f"Independent biological {table.level}; whole rows",
                "pipeline": "Refit feature filtering, imputation, scaling, PCA and BIC/GMM",
                "evaluation": "Project all original units through each fitted replicate pipeline",
                "oob": "Pair-specific requested/successful counts when both units were excluded",
                "intervals": "Failed-outcome identification bounds, not confidence intervals",
            },
            "interpretation": [
                "Candidate trait/environment groups; no delimited species or species probabilities",
                "Ecological groups remain at the declared level; no transfer to specimens",
                "Membership, bootstrap stability and significance are distinct evidence types",
                "Original-unit stability includes in-bag units; OOB denominators can be sparse",
                "Independence is unverified; related units and spatial dependence can violate it",
                "No categorical encoding, covariate adjustment, allometry correction or fusion",
                "Mean imputation assumes missingness is ignorable and can understate uncertainty",
            ],
            "provenance": {
                "configuration": configuration,
                "configuration_sha256": digest,
                "ingestion": loaded.summary["provenance"],
                "mixture_array_sha256": fingerprint_file(destination / "mixture.npz").sha256,
                "software": {
                    name: version(name)
                    for name in ("super-deli", "numpy", "scipy", "scikit-learn", "threadpoolctl")
                },
            },
        }
        if preprocessing_context is not None:
            summary["schema_version"] = 2
            summary["effective_preprocessing"] = preprocessing_context.declaration()
        write_json(summary, destination / "summary.json")
        return summary
    except Exception as exc:
        if preprocessing_context is not None:
            preprocessing_context.save(destination)
        write_json({"status": "failed", "error": str(exc)}, destination / "failure.json")
        raise
