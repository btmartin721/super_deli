"""Frozen full-covariance recovery diagnosis; no new significance or stability fits."""

import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath

from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.full_grouped_oracle import fit_oracle
from super_deli.benchmarking.full_grouped_power import retain_tests
from super_deli.benchmarking.grouped_recovery_inputs import (
    prepare_archived,
    replay,
    source_pins,
    stage_generated,
)
from super_deli.benchmarking.grouped_recovery_metrics import upstream_metrics
from super_deli.benchmarking.grouped_study import simulate_grouped
from super_deli.benchmarking.recovery_inputs import inventory
from super_deli.benchmarking.recovery_metrics import bic_terms, recovery_metrics
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import load_config, read_json, write_json
from super_deli.models.full_grouped_diagnostic_config import (
    COHORTS,
    PARTITIONS,
    FullGroupedDiagnosticConfig,
    variants,
)
from super_deli.models.grouped_config import GroupedStudyConfig


def source_check(source, config):
    import super_deli

    for name, digest in source_pins(config).items():
        if fingerprint_file(source / name).sha256 != digest:
            raise ValueError(f"Source fingerprint mismatch: {name}")
    if inventory(source / "study") != read_json(source / "study-audit/files.json"):
        raise ValueError("Source study inventory mismatch")
    study = GroupedStudyConfig.from_dict(read_json(source / "study.json"))
    if (
        study.n_units != config.small_n
        or study.n_datasets != config.n_datasets
        or study.calibration.schema_version != 2
    ):
        raise ValueError("Source cohort dimensions/schema disagree")
    variants(study.calibration.model)
    package = Path(super_deli.__file__).resolve().parent
    checked = []
    for name, digest in read_json(source / "runtime_sources.json").items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
            raise ValueError("Unsafe runtime source path")
        relative = path.relative_to("super_deli") if path.parts[0] == "super_deli" else path
        current = fingerprint_file(package / relative).sha256
        if relative.as_posix() not in {"__init__.py", "cli.py"} and current != digest:
            raise ValueError(f"Scientific source changed: {name}")
        checked.append(
            {"module": relative.as_posix(), "source_sha256": digest, "current_sha256": current}
        )
    if not any(c["module"] == "benchmarking/grouped_study.py" for c in checked):
        raise ValueError("Source generator fingerprint required")
    return study, checked


def metrics(run, labels, model):
    import numpy as np

    result = recovery_metrics(run, labels)
    result["bic_terms"] = bic_terms(
        read_json(run / "candidates.json"),
        result["selected_k"],
        len(labels),
        model.n_pcs,
        model.covariance_type,
    )
    result.update(upstream_metrics(run, labels))
    with np.load(run / "mixture.npz", allow_pickle=False) as arrays:
        eigenvalues = np.linalg.eigvalsh(arrays["covariances"])
        result["selected_covariance_eigenvalues"] = eigenvalues.tolist()
        result["selected_covariance_condition_numbers"] = (
            eigenvalues[:, -1] / eigenvalues[:, 0]
        ).tolist()
    return result


def _dataset(job):
    import numpy as np

    from super_deli.workflows.calibration_artifacts import fit_pipeline
    from super_deli.workflows.integrative import run_integrative

    root, row, model = job
    directory = root / row["cohort"] / f"dataset_{row['repetition']:04d}"
    truth = read_json(directory / "truth.json")
    labels = np.asarray(truth["labels"])
    declaration = load_config(directory / "raw/dataset.json")
    points = []
    for name, policy in variants(model).items():
        target = directory / name
        target.mkdir()
        record = fit_pipeline(
            target, declaration, directory / "raw", name, model=policy, runner=run_integrative
        )
        if record["status"] == "complete":
            if read_json(target / "run/summary.json")["unit_ids"] != truth["unit_ids"]:
                raise ValueError("Evaluation identity mismatch")
            record["metrics"] = metrics(target / "run", labels, policy)
        write_json(record, target / "record.json")
        points.append(
            {"cohort": row["cohort"], "repetition": row["repetition"], "arm": name, **record}
        )
    oracles = []
    for name in PARTITIONS:
        if points[0]["status"] == "complete":
            with np.load(directory / "baseline/run/mixture.npz", allow_pickle=False) as fitted:
                oracle = fit_oracle(
                    fitted["scores"], labels, name, model, directory / f"truth_{name}"
                )
        else:
            target = directory / f"truth_{name}"
            target.mkdir()
            oracle = {
                "status": "unattempted",
                "attempted": False,
                "partition": name,
                "reason": "baseline point failed",
            }
            write_json(oracle, target / "result.json")
        oracles.append({"cohort": row["cohort"], "repetition": row["repetition"], **oracle})
    return points, oracles


def accounting(rows, oracles, datasets, reproduced):
    return {
        "schema_version": 1,
        "datasets": datasets,
        "planned_points": len(rows),
        "successful_points": sum(r["status"] == "complete" for r in rows),
        "failed_points": sum(r["status"] == "failed" for r in rows),
        "planned_truth_initialized": len(oracles),
        "successful_truth_initialized": sum(r["status"] == "complete" for r in oracles),
        "failed_truth_initialized": sum(r["status"] == "failed" for r in oracles),
        "unattempted_truth_initialized": sum(r["status"] == "unattempted" for r in oracles),
        "replayed_archived": sum(r["status"] == "reproduced" for r in reproduced),
        "bootstrap_or_calibration_fits": 0,
    }


def run_full_grouped_diagnostics(config, *, source_artifacts, output_dir, workers=1):
    config = FullGroupedDiagnosticConfig.from_dict(config.to_dict())
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use 1..4 independent one-thread workers")
    root, source = Path(output_dir).resolve(), Path(source_artifacts).resolve()
    if root == source or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Diagnostic and immutable source must be separate")
    if root.exists():
        raise FileExistsError(f"Output already exists: {root}")
    study, checked = source_check(source, config)
    root.mkdir()
    for original, name in (
        ("study.json", "source_declaration.json"),
        ("study-audit/files.json", "source_inventory.json"),
        ("runtime_sources.json", "source_runtime.json"),
    ):
        shutil.copyfile(source / original, root / name)
    write_json(config.to_dict(), root / "configuration.json")
    write_json(study.to_dict(), root / "source_study.json")
    write_json(
        {k: m.to_dict() for k, m in variants(study.calibration.model).items()}, root / "grid.json"
    )
    write_json(
        {
            "source_pins": source_pins(config),
            "compatible_scientific_modules": checked,
            "workers": workers,
            "threads_per_job": 1,
            "replacement_fits": 0,
            "truth_role": "Separate optimization diagnosis; never unsupervised selection",
        },
        root / "execution.json",
    )
    write_json(retain_tests(source, root / "retained_tests", study), root / "retained_tails.json")
    for cohort in COHORTS:
        (root / cohort).mkdir()
    archived = prepare_archived(source, root, study)
    for i in range(config.n_datasets):
        shutil.copyfile(
            source / "study/complementary" / f"dataset_{i:04d}/calibration/archive_inventory.json",
            root / "source_snapshots" / f"dataset_{i:04d}/archive_inventory.json",
        )
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
    jobs = [(root, row, study.calibration.model) for row in datasets]
    rows, oracles = [], []
    if workers == 1:
        for points, oracle in map(_dataset, jobs):
            rows.extend(points)
            oracles.extend(oracle)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for points, oracle in executor.map(_dataset, jobs):
                rows.extend(points)
                oracles.extend(oracle)
    write_json(rows, root / "points.json")
    write_json(oracles, root / "truth_initialized.json")
    reproduced = replay(root, config.n_datasets)
    write_json(reproduced, root / "replay.json")
    summary = accounting(rows, oracles, len(datasets), reproduced)
    write_json(summary, root / "summary.json")
    write_json(inventory(root), root / "files.json")
    return summary
