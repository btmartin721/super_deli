"""Fresh raw-data generation, ingestion and complete point refits for every null draw."""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from super_deli.benchmarking.config import derived_seed
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.results import Evidence
from super_deli.core.schema import config_from_dict
from super_deli.io.config import read_json, write_json
from super_deli.models.pipeline_config import PipelineCalibrationConfig
from super_deli.workflows.calibration_artifacts import archive_tree as _archive_tree
from super_deli.workflows.calibration_artifacts import fit_pipeline

SCOPE = (
    "Global Monte Carlo reference-population test under a fully declared raw-data simulation law. "
    "Every draw refits ingestion, encoding/filtering/imputation, GPA, view PCA, aggregation, "
    "fusion, joint PCA and the configured GMM search. Weights, dimensions, hyperparameters "
    "and analysis seed are prespecified. No empirical null fitting, evolutionary species null, "
    "LD/spatial process or cluster-specific significance is supplied. Source counts and baseline "
    "locus frequencies are redrawn under the reference law; calibration is marginal over them."
)


TAIL_OPTIONS = {
    "p_value_kind": "Monte Carlo rank test under a declared raw-data reference population",
    "decision_key": "reject_reference_population",
    "interval_scope": (
        "Binomial reference-tail interval conditional on the observed statistic; "
        "excludes null-law misspecification and is not species confidence; "
        "failure bounds are not confidence intervals"
    ),
}


def _dependencies():
    try:
        from super_deli.benchmarking.simulation import simulate, write_simulation
        from super_deli.io.genomic import _dependencies as genomic_dependencies
        from super_deli.models.gaussian_calibration import bic_improvement
        from super_deli.models.monte_carlo import summarize_upper_tail
        from super_deli.workflows.integrative import run_integrative

        genomic_dependencies()
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Install 'super-deli[clustering,genomics]' for full-pipeline calibration"
        ) from exc
    return simulate, write_simulation, bic_improvement, summarize_upper_tail, run_integrative


def _compatible(run, config, directory):
    import numpy as np

    reference = config.reference
    if len(run["unit_ids"]) > reference.n_units:
        raise ValueError("Observed target count exceeds the declared reference design")
    if not reference.view_missing_rate and len(run["unit_ids"]) != reference.n_units:
        raise ValueError("Observed target count differs from the complete reference design")
    for i, view in enumerate(run["views"]):
        with (directory / f"view_{i:03d}/source_values.npy").open("rb") as handle:
            shape = np.load(handle, allow_pickle=False).shape
        if view["kind"] == "genomic" and shape[1:] != (reference.n_loci,):
            raise ValueError("Observed locus count differs from the reference design")
        if view["kind"] == "morphometric" and shape[1:] != (6, 2):
            raise ValueError("Reference shapes require six 2D fixed landmarks")
        kind = view["kind"]
        expected = 5 if kind == "morphological" else 4
        if kind in {"morphological", "ecological"} and (
            list(view["feature_names"]) != [f"feature_{i}" for i in range(expected)]
            or list(view["feature_types"])
            != ["continuous"] * 4 + (["count"] if expected == 5 else [])
        ):
            raise ValueError("Observed feature schema differs from the reference population")
        if kind == "morphometric" and list(view["landmark_ids"]) != [f"L{i + 1}" for i in range(6)]:
            raise ValueError("Observed landmarks differ from the six-point reference template")
        sizes = [len(group["observation_ids"]) for group in view["members"]]
        if kind == "ecological" or reference.linkage == "matched":
            valid = all(n == 1 for n in sizes)
        else:
            valid = (
                all(1 <= n <= reference.max_group_members for n in sizes)
                if reference.unequal_groups
                else all(n == 2 for n in sizes)
            )
        if not valid:
            raise ValueError("Observed source counts differ from the reference design")


def run_reference_calibration(dataset, *, config, base_dir, output_dir):
    config = PipelineCalibrationConfig.from_dict(config.to_dict())
    dataset = config_from_dict(dataset.to_dict())
    simulate, write_simulation, _, summarize, run_integrative = _dependencies()
    source = Path(base_dir).resolve()
    destination = Path(output_dir).resolve()
    # Outputs can be siblings of input files, but never rewrite a declared input.
    from super_deli.core.validation import input_paths

    inputs = tuple(input_paths(dataset, source))
    if any(p.resolve().is_relative_to(destination) for p in inputs):
        raise ValueError("Output would contain an observed input; choose a separate directory")
    observed_hashes = {str(p): fingerprint_file(p).sha256 for p in inputs}
    destination.mkdir(exist_ok=False)
    write_json(config.to_dict(), destination / "configuration.json")
    inventory, records = {}, []
    from super_deli.benchmarking import simulation as generator

    generator_sha = fingerprint_file(generator.__file__).sha256
    zip_path = destination / "pipeline_artifacts.zip"

    def fit_dataset(staging, declaration, directory, prefix):
        record = fit_pipeline(
            staging,
            declaration,
            directory,
            prefix,
            model=config.model,
            runner=run_integrative,
            validate=lambda run, path: _compatible(run, config, path),
        )
        _archive_tree(archive, staging, prefix, inventory, config.max_archive_bytes)
        return record

    try:
        with ZipFile(zip_path, "x", compression=ZIP_DEFLATED, compresslevel=4) as archive:
            with TemporaryDirectory(prefix="observed-", dir=destination) as temporary:
                point = fit_dataset(Path(temporary), dataset, source, "observed")
            write_json(point, destination / "observed.json")
            if point["status"] == "complete":
                for i in range(config.n_simulations):
                    seed = derived_seed(config.random_seed, config.reference.law, i, "raw_null")
                    with TemporaryDirectory(prefix="null-", dir=destination) as temporary:
                        staging = Path(temporary)
                        data = simulate(config.reference.scenario(), config.reference, seed)
                        write_simulation(data, staging / "simulation")
                        declaration = config_from_dict(
                            read_json(staging / "simulation/observed/dataset.json")
                        )
                        record = fit_dataset(
                            staging, declaration, staging / "simulation/observed", f"null_{i:04d}"
                        )
                    record.update(replicate=i, simulation_seed=seed)
                    records.append(record)
                    write_json(record, destination / f"draw_{i:04d}.json")
        if observed_hashes != {str(p): fingerprint_file(p).sha256 for p in inputs}:
            raise ValueError("Observed inputs changed during calibration")
        write_json(inventory, destination / "archive_inventory.json")
        write_json(records, destination / "null_replicates.json")
        inference = (
            summarize(
                point["statistic"],
                records,
                alpha=config.alpha,
                **TAIL_OPTIONS,
            )
            if point["status"] == "complete"
            else {
                "status": "point_failed",
                "requested": config.n_simulations,
                "successful": 0,
                "failed": 0,
                "p_value": None,
                "reject_reference_population": None,
            }
        )
        evidence = Evidence(
            "p_value",
            "declared_raw_reference_population",
            "available" if inference["p_value"] is not None else "unavailable",
            method="Full-pipeline Monte Carlo BIC-improvement rank test",
            value=inference["p_value"],
            replicates=config.n_simulations,
            null_hypothesis=(
                f"Observed data follow {config.reference.law} "
                "with the declared design and parameters"
            ),
            multiplicity=(
                "none across analyses; one prespecified global statistic includes the full K search"
            ),
            reason=None
            if inference["p_value"] is not None
            else "A point or requested null pipeline fit failed",
        )
        write_json(asdict(evidence), destination / "evidence.json")
        summary = {
            "schema_version": 1,
            "status": "complete",
            "scope": SCOPE,
            "configuration_sha256": configuration_hash(config.to_dict()),
            "generator_sha256": generator_sha,
            "observed_configuration": dataset.to_dict(),
            "observed_input_sha256": observed_hashes,
            "analysis_seed_policy": (
                "Identical prespecified model.random_seed for observed and all null fits"
            ),
            "point": point,
            "inference": inference,
            "planned_fits": config.n_simulations + 1,
            "attempted_fits": len(records) + 1,
            "successful_fits": sum(r["status"] == "complete" for r in (point, *records)),
            "failed_fits": sum(r["status"] == "failed" for r in (point, *records)),
            "unattempted_fits": config.n_simulations - len(records),
            "artifacts": {
                name: fingerprint_file(destination / name).sha256
                for name in (
                    "pipeline_artifacts.zip",
                    "archive_inventory.json",
                    "null_replicates.json",
                    "evidence.json",
                )
            },
            "software": {
                name: version(name)
                for name in (
                    "super-deli",
                    "snpio",
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "threadpoolctl",
                )
            },
        }
        write_json(summary, destination / "summary.json")
        return summary
    except Exception as exc:
        write_json({"status": "failed", "error": str(exc)}, destination / "failure.json")
        raise
