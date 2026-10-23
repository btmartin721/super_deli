"""Frozen four-arm grouped recovery study with attached whole-unit stability."""

import shutil
from concurrent.futures import ProcessPoolExecutor
from importlib import import_module
from pathlib import Path, PurePosixPath

from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.grouped_recovery_metrics import recovery_metrics
from super_deli.benchmarking.grouped_study import simulate_grouped
from super_deli.benchmarking.recovery_inputs import inventory
from super_deli.benchmarking.shared_study import accounting, point_metrics
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import load_config, read_json, write_json
from super_deli.models.grouped_config import GroupedStudyConfig
from super_deli.models.grouped_pca_study_config import (
    ARMS,
    COHORTS,
    PINS,
    GroupedPCAStudyConfig,
    study_model,
)

# Changes necessary for the approved new option; every other archived module must match.
SOURCE_CHANGES = {
    "__init__.py",
    "cli.py",
    "models/integrative_config.py",
    "models/integrative.py",
    "models/grouped_config.py",
    "models/pipeline_config.py",
    "models/empirical_config.py",
    "models/joint_config.py",
    "io/integrative_artifacts.py",
    "workflows/integrative.py",
    "workflows/score_calibration.py",
}


def source_check(source, config):
    import super_deli

    for field, name in PINS.items():
        if fingerprint_file(source / name).sha256 != getattr(config, field):
            raise ValueError(f"Source fingerprint mismatch: {name}")
    actual = inventory(source / "study")
    actual.pop("files.json")
    if actual != read_json(source / "study/files.json"):
        raise ValueError("Source diagnostic inventory mismatch")
    prior = GroupedStudyConfig.from_dict(read_json(source / "study/source_study.json"))
    if (
        prior.n_datasets != config.n_datasets
        or prior.n_units != config.small_n
        or prior.calibration.model != config.model
    ):
        raise ValueError("Source dimensions or baseline configuration differ")
    package = Path(super_deli.__file__).resolve().parent
    checked = []
    for name, digest in read_json(source / "runtime_sources.json").items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
            raise ValueError("Unsafe scientific source path")
        relative = path.relative_to("super_deli") if path.parts[0] == "super_deli" else path
        current = fingerprint_file(package / relative).sha256
        changed = current != digest
        if changed and relative.as_posix() not in SOURCE_CHANGES:
            raise ValueError(f"Unapproved scientific source change: {relative}")
        checked.append(
            {
                "module": relative.as_posix(),
                "source_sha256": digest,
                "current_sha256": current,
                "changed": changed,
            }
        )
    if not any(p["module"] == "benchmarking/grouped_study.py" for p in checked):
        raise ValueError("Source generator fingerprint required")
    return prior, checked


def simulate(cohort, n, seed, *, target_level):
    import numpy as np

    case = (
        cohort
        if cohort in {"independent_gaussian", "correlated_gaussian"}
        else "correlated_gaussian"
        if cohort == "cline"
        else "complementary"
    )
    values, layout, labels = simulate_grouped(case, n, seed, target_level=target_level)
    targets = np.repeat(np.arange(n), layout["counts"])
    loadings = np.array([1, 0.7, 0.5, 0.3])
    if cohort == "unequal_variance":
        a, b = 2 * (labels // 2) - 1, 2 * (labels % 2) - 1
        mean = np.linspace(-2, 2, 8)
        centers = np.column_stack((a[:, None] * loadings, b[:, None] * loadings)) * 3 + mean
        factors = np.array([0.5, 0.8, 1.2, 1.8])[labels]
        values["morphological"] = (
            centers[targets, :4]
            + (values["morphological"] - centers[targets, :4]) * factors[targets, None]
        )
        values["ecological"] = (
            centers[:, 4:] + (values["ecological"] - centers[:, 4:]) * factors[:, None]
        )
    elif cohort == "cline":
        coordinate = np.random.default_rng(derived_seed(seed, "cline", 0, "gradient")).uniform(
            -1, 1, n
        )
        values["morphological"] += 3 * coordinate[targets, None] * loadings
        values["ecological"] += 3 * coordinate[:, None] * loadings
    return values, layout, labels


def stage(values, layout, labels, directory, has_groups):
    import numpy as np

    from super_deli.models.grouped_null import write_raw_views

    directory.mkdir()
    write_raw_views(values, layout, directory / "raw")
    with (directory / "values.npz").open("xb") as handle:
        np.savez(handle, **values)
    write_json(layout, directory / "layout.json")
    write_json(
        {
            "unit_ids": layout["unit_ids"],
            "labels": labels.tolist(),
            "has_groups": has_groups,
            "scope": "Generating truth for evaluation only; not species",
        },
        directory / "truth.json",
    )


def _dataset(job):
    import numpy as np

    from super_deli.workflows.calibration_artifacts import fit_pipeline
    from super_deli.workflows.integrative import run_integrative

    root, config, declaration = job
    cohort, i = declaration["cohort"], declaration["repetition"]
    directory = root / cohort / f"dataset_{i:04d}"
    dataset = load_config(directory / "raw/dataset.json")
    labels = np.asarray(read_json(directory / "truth.json")["labels"])
    records = []
    for arm in ARMS:
        model = study_model(config, cohort, i, arm)
        output = directory / arm
        output.mkdir()
        record = fit_pipeline(
            output, dataset, directory / "raw", arm, model=model, runner=run_integrative
        )
        replicas = []
        if record["status"] == "complete":
            run = read_json(output / "run/summary.json")
            if run["unit_ids"] != read_json(directory / "truth.json")["unit_ids"]:
                raise ValueError("Evaluation identity mismatch")
            record["metrics"] = (
                recovery_metrics(output / "run", labels)
                if declaration["has_groups"]
                else point_metrics(output / "run", labels, False)
            )
            replicas = read_json(output / "run/bootstrap_replicates.json")
        record["bootstrap"] = {
            "requested": model.n_bootstrap,
            "attempted": len(replicas),
            "successful": sum(r["status"] == "complete" for r in replicas),
            "failed": sum(r["status"] == "failed" for r in replicas),
        }
        write_json(record, output / "record.json")
        records.append({"cohort": cohort, "repetition": i, "arm": arm, **record})
    return records


def replay(root, count):
    import numpy as np

    records = []
    for i in range(count):
        directory = root / "archived" / f"dataset_{i:04d}"
        old, new = (
            read_json(directory / name / "record.json") for name in ("source_baseline", "source_2")
        )
        if new["status"] == "failed":
            records.append({"repetition": i, "status": "failed", "error": new["error"]})
            continue
        if old["selected_k"] != new["selected_k"]:
            raise ValueError("Archived baseline selection mismatch")
        for a, b in zip(old["candidates"], new["candidates"], strict=True):
            if {k: v for k, v in a.items() if k != "bic"} != {
                k: v for k, v in b.items() if k != "bic"
            }:
                raise ValueError("Archived candidate constraint mismatch")
            if a["bic"] is not None:
                np.testing.assert_allclose(a["bic"], b["bic"], atol=1e-7, rtol=1e-10)
        with (
            np.load(directory / "source_baseline/run/mixture.npz", allow_pickle=False) as a,
            np.load(directory / "source_2/run/mixture.npz", allow_pickle=False) as b,
        ):
            np.testing.assert_allclose(a["scores"], b["scores"], atol=1e-11, rtol=1e-9)
            np.testing.assert_array_equal(a["labels"], b["labels"])
        records.append({"repetition": i, "status": "reproduced"})
    return records


def run_grouped_pca_study(config, *, source_artifacts, output_dir, workers=1):
    for dependency in ("numpy", "scipy", "sklearn"):
        import_module(dependency)
    config = GroupedPCAStudyConfig.from_dict(config.to_dict())
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use 1..4 independent single-thread workers")
    root, source = Path(output_dir).resolve(), Path(source_artifacts).resolve()
    if root == source or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Study must be separate from immutable source artifacts")
    if root.exists():
        raise FileExistsError(f"Output already exists: {root}")
    prior, checked = source_check(source, config)
    root.mkdir()
    write_json(config.to_dict(), root / "configuration.json")
    write_json(prior.to_dict(), root / "source_study.json")
    write_json(
        {arm: study_model(config, "archived", 0, arm).to_dict() for arm in ARMS}, root / "grid.json"
    )
    write_json(
        {
            "workers": workers,
            "compatible_sources": checked,
            "unequal_variance_sd_multipliers": [0.5, 0.8, 1.2, 1.8],
            "source_artifact_id": "grouped-recovery-20261023",
            "source_files_sha256": config.source_inventory_sha256,
        },
        root / "execution.json",
    )
    declarations = []
    for cohort in COHORTS:
        (root / cohort).mkdir()
        for i in range(config.n_datasets):
            n = config.large_n if cohort == "fresh_256" else config.small_n
            has_groups = cohort in {"archived", "fresh_64", "fresh_256", "unequal_variance"}
            seed = (
                derived_seed(prior.random_seed, "complementary", i, "observed_data")
                if cohort == "archived"
                else derived_seed(config.random_seed, cohort, i, "observed_data")
            )
            directory = root / cohort / f"dataset_{i:04d}"
            values, layout, labels = simulate(
                cohort, n, seed, target_level=config.model.target_level
            )
            stage(values, layout, labels, directory, has_groups)
            if cohort == "archived":
                original = source / "study/archived" / directory.name
                if inventory(original / "raw") != inventory(directory / "raw"):
                    raise ValueError("Archived raw data/identities differ")
                shutil.copytree(original / "baseline", directory / "source_baseline")
            declarations.append(
                {"cohort": cohort, "repetition": i, "n": n, "seed": seed, "has_groups": has_groups}
            )
    write_json(declarations, root / "datasets.json")
    jobs = [(root, config, row) for row in declarations]
    if workers == 1:
        records = [r for job in jobs for r in _dataset(job)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            records = [r for result in executor.map(_dataset, jobs) for r in result]
    write_json(records, root / "points.json")
    reproduced = replay(root, config.n_datasets)
    write_json(reproduced, root / "replay.json")
    summary = {
        "schema_version": 1,
        "datasets": len(declarations),
        "reproduced_archived": sum(r["status"] == "reproduced" for r in reproduced),
        **accounting(records),
    }
    write_json(summary, root / "summary.json")
    write_json(inventory(root), root / "files.json")
    return summary
