"""Approximate fitted-null bootstrap with complete continuous-view pipeline refits."""

from __future__ import annotations

from dataclasses import asdict
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from super_deli.benchmarking.config import derived_seed
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.results import Evidence
from super_deli.core.schema import config_from_dict
from super_deli.core.validation import input_paths
from super_deli.io.config import write_json
from super_deli.models.empirical_config import EmpiricalCalibrationConfig
from super_deli.workflows.calibration_artifacts import archive_tree, fit_counts, fit_pipeline

SCOPE = (
    "Approximate plug-in joint Gaussian null fitted to matched complete continuous morphology/"
    "ecology before PCA. Each draw redoes ingestion, preprocessing, view PCA, block fusion, joint "
    "PCA and the full GMM search. Fitted nuisance parameters remove the known-law exact-rank "
    "guarantee. No genomic/count/shape/grouped null, species-boundary or cluster-specific test, "
    "spatial/related sampling model or selection across configurations is supplied."
)
TAIL_OPTIONS = {
    "p_value_kind": "Approximate fitted raw-Gaussian parametric bootstrap with plus-one rank",
    "decision_key": "reject_joint_raw_gaussian",
    "interval_scope": (
        "Binomial fitted-null tail interval conditional on observed T and fitted parameters; "
        "excludes null-estimation/misspecification uncertainty, not species confidence; "
        "failed-outcome bounds are not confidence intervals"
    ),
}


def _dependencies():
    try:
        import numpy as np

        import_module("sklearn")

        from super_deli.models.empirical_null import fit_null, simulate_null, write_raw_views
        from super_deli.models.monte_carlo import summarize_upper_tail
        from super_deli.workflows.integrative import run_integrative
    except (ImportError, OSError) as exc:
        raise RuntimeError("Install 'super-deli[clustering]' for empirical calibration") from exc
    return np, fit_null, simulate_null, write_raw_views, summarize_upper_tail, run_integrative


def run_empirical_calibration(dataset, *, config, base_dir, output_dir):
    config = EmpiricalCalibrationConfig.from_dict(config.to_dict())
    return run_fitted_calibration(
        dataset,
        config=config,
        base_dir=base_dir,
        output_dir=output_dir,
        dependencies=_dependencies(),
        scope=SCOPE,
        tail_options=TAIL_OPTIONS,
        seed_namespace="empirical_raw_gaussian_v1",
        target="joint_gaussian_raw_continuous_views",
        null_hypothesis=(
            "Matched continuous rows follow one joint Gaussian with unknown mean/covariance"
        ),
    )


def run_fitted_calibration(
    dataset,
    *,
    config,
    base_dir,
    output_dir,
    dependencies,
    scope,
    tail_options,
    seed_namespace,
    target,
    null_hypothesis,
):
    """Shared execution and failure accounting; each caller supplies its raw null."""
    dataset = config_from_dict(dataset.to_dict())
    np, fit_null, simulate_null, write_raw_views, summarize, runner = dependencies
    decision_key = tail_options["decision_key"]
    from super_deli.io.tabular import ingest_tabular

    source, destination = Path(base_dir).resolve(), Path(output_dir).resolve()
    inputs = tuple(input_paths(dataset, source))
    if any(p.resolve().is_relative_to(destination) for p in inputs):
        raise ValueError("Output would contain an observed input; choose a separate directory")
    hashes = {str(p): fingerprint_file(p).sha256 for p in inputs}
    destination.mkdir(exist_ok=False)
    write_json(config.to_dict(), destination / "configuration.json")
    point, records, inventory = None, [], {}
    try:
        loaded = ingest_tabular(
            dataset,
            base_dir=source,
            output_dir=destination / "null_ingestion",
            modality_names=[v.modality for v in config.model.views],
        )
        null = fit_null(dataset, loaded.tables, config)
        with (destination / "null_model.npz").open("xb") as handle:
            np.savez(handle, **null.model_arrays())
        write_json(null.layout, destination / "layout.json")
        with ZipFile(
            destination / "pipeline_artifacts.zip", "x", compression=ZIP_DEFLATED, compresslevel=4
        ) as archive:
            with TemporaryDirectory(prefix="observed-", dir=destination) as temporary:
                staging = Path(temporary)
                point = fit_pipeline(
                    staging, dataset, source, "observed", model=config.model, runner=runner
                )
                write_json(point, destination / "observed.json")
                archive_tree(archive, staging, "observed", inventory, config.max_archive_bytes)
            if point["status"] == "complete":
                for i in range(config.n_simulations):
                    seed = derived_seed(config.random_seed, seed_namespace, i, "null_data")
                    with TemporaryDirectory(prefix="null-", dir=destination) as temporary:
                        staging = Path(temporary)
                        values = simulate_null(null, seed)
                        declaration = write_raw_views(values, null.layout, staging / "raw")
                        record = fit_pipeline(
                            staging,
                            declaration,
                            staging / "raw",
                            f"null_{i:04d}",
                            model=config.model,
                            runner=runner,
                        )
                        record.update(replicate=i, simulation_seed=seed)
                        records.append(record)
                        write_json(record, destination / f"draw_{i:04d}.json")
                        archive_tree(
                            archive,
                            staging,
                            record["archive_prefix"],
                            inventory,
                            config.max_archive_bytes,
                        )
        if hashes != {str(p): fingerprint_file(p).sha256 for p in inputs}:
            raise ValueError("Observed inputs changed during calibration")
        write_json(inventory, destination / "archive_inventory.json")
        write_json(records, destination / "null_replicates.json")
        inference = (
            summarize(point["statistic"], records, alpha=config.alpha, **tail_options)
            if point["status"] == "complete"
            else {
                "status": "point_failed",
                "requested": config.n_simulations,
                "successful": 0,
                "failed": 0,
                "p_value": None,
                decision_key: None,
            }
        )
        evidence = Evidence(
            "p_value",
            target,
            "available" if inference["p_value"] is not None else "unavailable",
            method="Approximate full-pipeline fitted-Gaussian BIC-improvement bootstrap",
            value=inference["p_value"],
            replicates=config.n_simulations,
            null_hypothesis=null_hypothesis,
            multiplicity=(
                "none across analyses; one prespecified global statistic includes the full K search"
            ),
            reason=None
            if inference["p_value"] is not None
            else "A requested point or null fit failed",
        )
        write_json(asdict(evidence), destination / "evidence.json")
        result = {
            "schema_version": 1,
            "status": "complete",
            "scope": scope,
            "configuration_sha256": configuration_hash(config.to_dict()),
            "observed_configuration": dataset.to_dict(),
            "observed_input_sha256": hashes,
            "null_fit": null.layout["fit"],
            "n_units": len(null.values),
            "n_raw_features": null.values.shape[1],
            "analysis_seed_policy": (
                "Identical prespecified model.random_seed for observed and all null fits"
            ),
            "point": point,
            "inference": inference,
            **fit_counts(point, records, config.n_simulations),
            "artifacts": {
                name: fingerprint_file(destination / name).sha256
                for name in (
                    "null_model.npz",
                    "layout.json",
                    "pipeline_artifacts.zip",
                    "archive_inventory.json",
                    "null_replicates.json",
                    "evidence.json",
                )
            },
            "software": {
                name: version(name)
                for name in ("super-deli", "numpy", "scipy", "scikit-learn", "threadpoolctl")
            },
        }
        write_json(result, destination / "summary.json")
        return result
    except Exception as exc:
        write_json(
            {
                "status": "failed",
                "error": str(exc),
                **fit_counts(point, records, config.n_simulations),
            },
            destination / "failure.json",
        )
        raise
