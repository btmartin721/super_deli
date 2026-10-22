"""Ingest landmarks and export auditable candidate shape clusters and stability."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

from super_deli.core.data import DatasetConfig
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.results import DelimitationResult
from super_deli.io.config import read_json, write_json
from super_deli.models.shape_cluster_config import ShapeClusterConfig
from super_deli.workflows.cluster_artifacts import candidate_result, write_cluster_artifacts
from super_deli.workflows.shape_pca import run_shape_pca


def _result(dataset, fit, digest, level):
    return candidate_result(
        dataset,
        fit,
        digest,
        level,
        model_name="shape_pca_full_gmm",
        space="shape PCA",
        pipeline="GPA/PCA/GMM",
        result_type=DelimitationResult,
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
        write_cluster_artifacts(
            fit,
            result,
            scores=fit.shape.projection.scores,
            observation_ids=[r.observation_id for r in records],
            directory=destination,
        )
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
