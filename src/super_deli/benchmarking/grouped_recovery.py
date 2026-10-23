"""Grouped point-pipeline diagnostics, with no bootstrap or significance refits."""

from concurrent.futures import ProcessPoolExecutor
from importlib import import_module
from pathlib import Path

from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.grouped_recovery_inputs import (
    check_sources,
    prepare_archived,
    replay,
    stage_generated,
)
from super_deli.benchmarking.grouped_recovery_metrics import recovery_metrics, truth_initialized
from super_deli.benchmarking.grouped_study import simulate_grouped
from super_deli.benchmarking.recovery_inputs import inventory
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import load_config, read_json, write_json
from super_deli.models.grouped_recovery_config import COHORTS, GroupedRecoveryConfig, variants


def _dataset(job):
    import numpy as np

    from super_deli.workflows.calibration_artifacts import fit_pipeline
    from super_deli.workflows.integrative import run_integrative

    root, cohort, i, model = job
    directory = root / cohort / f"dataset_{i:04d}"
    truth = read_json(directory / "truth.json")
    labels = np.asarray(truth["labels"])
    rows = []
    for name, policy in variants(model).items():
        raw = directory / ("first_raw" if name == "first_specimen" else "raw")
        declaration = load_config(raw / "dataset.json")
        staging = directory / name
        staging.mkdir()
        record = fit_pipeline(
            staging, declaration, raw, "point", model=policy, runner=run_integrative
        )
        if record["status"] == "complete":
            if read_json(staging / "run/summary.json")["unit_ids"] != truth["unit_ids"]:
                raise ValueError("Point output and evaluation identities disagree")
            record["metrics"] = recovery_metrics(staging / "run", labels)
        write_json(record, staging / "record.json")
        rows.append({"cohort": cohort, "repetition": i, "variant": name, **record})
    baseline = rows[0]
    if baseline["status"] == "complete":
        with np.load(directory / "baseline/run/mixture.npz", allow_pickle=False) as arrays:
            oracle = truth_initialized(
                arrays["scores"], labels, model, directory / "truth_initialized"
            )
    else:
        oracle = {"status": "unattempted", "attempted": False, "reason": "baseline point failed"}
        (directory / "truth_initialized").mkdir()
        write_json(oracle, directory / "truth_initialized/result.json")
    return rows, {"cohort": cohort, "repetition": i, **oracle}


def run_grouped_recovery(config, *, source_artifacts, output_dir, workers=1):
    try:
        for dependency in ("numpy", "scipy", "sklearn"):
            import_module(dependency)
    except (ImportError, OSError) as exc:
        raise RuntimeError("Install 'super-deli[clustering]' for recovery diagnostics") from exc
    config = GroupedRecoveryConfig.from_dict(config.to_dict())
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four independent single-thread workers")
    root, source = Path(output_dir).resolve(), Path(source_artifacts).resolve()
    if root == source or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Diagnostic artifacts must be separate from source artifacts")
    if root.exists():
        raise FileExistsError(f"Output already exists: {root}")
    study, pins, checked = check_sources(source, config)
    grid = variants(study.calibration.model)
    root.mkdir()
    # Write settings and source fingerprints BEFORE preparing any fresh dataset.
    write_json(config.to_dict(), root / "configuration.json")
    write_json(study.to_dict(), root / "source_study.json")
    write_json({name: model.to_dict() for name, model in grid.items()}, root / "grid.json")
    write_json(
        {"source_pins": pins, "compatible_scientific_modules": checked, "workers": workers},
        root / "execution.json",
    )
    for cohort in COHORTS:
        (root / cohort).mkdir()
    archived = prepare_archived(source, root, study)
    write_json(archived, root / "source_evidence.json")
    datasets = []
    for cohort in COHORTS:
        n = config.large_n if cohort == "fresh_256" else config.small_n
        for i in range(config.n_datasets):
            seed = (
                archived[i]["seed"]
                if cohort == "archived"
                else derived_seed(config.random_seed, cohort, i, "observed_data")
            )
            if cohort != "archived":
                stage_generated(
                    *simulate_grouped(
                        "complementary", n, seed, target_level=study.calibration.model.target_level
                    ),
                    root / cohort / f"dataset_{i:04d}",
                )
            datasets.append({"cohort": cohort, "repetition": i, "n": n, "seed": seed})
    write_json(datasets, root / "datasets.json")
    jobs = [(root, row["cohort"], row["repetition"], study.calibration.model) for row in datasets]
    rows, oracles = [], []
    if workers == 1:
        outcomes = map(_dataset, jobs)
        for point_rows, oracle in outcomes:
            rows.extend(point_rows)
            oracles.append(oracle)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for point_rows, oracle in executor.map(_dataset, jobs):
                rows.extend(point_rows)
                oracles.append(oracle)
    write_json(rows, root / "points.json")
    write_json(oracles, root / "truth_initialized.json")
    reproduced = replay(root, config.n_datasets)
    write_json(reproduced, root / "replay.json")
    summary = {
        "schema_version": 1,
        "datasets": len(datasets),
        "planned_points": len(jobs) * len(grid),
        "successful_points": sum(r["status"] == "complete" for r in rows),
        "failed_points": sum(r["status"] == "failed" for r in rows),
        "planned_truth_initialized": len(jobs),
        "successful_truth_initialized": sum(r["status"] == "complete" for r in oracles),
        "failed_truth_initialized": sum(r["status"] == "failed" for r in oracles),
        "unattempted_truth_initialized": sum(r["status"] == "unattempted" for r in oracles),
        "replayed_archived": sum(r["status"] == "reproduced" for r in reproduced),
        "bootstrap_or_calibration_fits": 0,
    }
    write_json(summary, root / "summary.json")
    files = inventory(root)
    write_json(files, root / "files.json")
    return {**summary, "inventory_sha256": fingerprint_file(root / "files.json").sha256}
