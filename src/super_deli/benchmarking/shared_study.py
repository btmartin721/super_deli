"""Truth-separated paired covariance-family study with a fixed stability subset."""

import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from importlib import import_module
from pathlib import Path

from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.joint_study import simulate_joint
from super_deli.benchmarking.recovery_inputs import inventory, stage_generated
from super_deli.benchmarking.recovery_metrics import axis_r2, bic_terms
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import load_config, read_json, write_json
from super_deli.models.joint_config import JointStudyConfig
from super_deli.models.shared_config import COHORTS, FAMILIES, SharedStudyConfig


def study_seed(config, cohort, i):
    return derived_seed(config.random_seed, cohort, i, "observed_data")


def study_model(config, cohort, i, family):
    draws = (
        config.stability_draws
        if cohort in {"fresh_64", "fresh_256"} and i < config.stability_datasets
        else 0
    )
    return replace(config.model, covariance_type=family, n_bootstrap=draws)


def simulate_shared(cohort, n, seed):
    import numpy as np

    case = cohort if cohort in {"zero_association", "cline"} else "complementary"
    values, layout, labels = simulate_joint(case, n, seed)
    if cohort == "unequal_variance":
        a, b = 2 * (labels // 2) - 1, 2 * (labels % 2) - 1
        centers = 3 * np.column_stack(
            [a[:, None] * np.array([1, 0.7, 0.5, 0.3]), b[:, None] * np.array([1, 0.7, 0.5, 0.3])]
        )
        # Prespecified residual SD multipliers; this violates shared covariance.
        values["phenotypes"] = (
            centers
            + (values["phenotypes"] - centers) * np.array([0.5, 0.8, 1.2, 1.8])[labels, None]
        )
    return values, layout, labels


def source_check(root, config):
    pins = {
        "study/files.json": config.source_inventory_sha256,
        "study/summary.json": config.source_summary_sha256,
        "study/source_study.json": config.source_study_sha256,
    }
    for name, digest in pins.items():
        if fingerprint_file(root / name).sha256 != digest:
            raise ValueError(f"Source fingerprint mismatch: {name}")
    expected = read_json(root / "study/files.json")
    actual = inventory(root / "study")
    actual.pop("files.json")
    if actual != expected:
        raise ValueError("Source diagnostic inventory mismatch")
    prior = JointStudyConfig.from_dict(read_json(root / "study/source_study.json"))
    if (
        prior.n_datasets != config.n_datasets
        or prior.n_units != config.small_n
        or prior.calibration.model.to_dict() != config.model.to_dict()
    ):
        raise ValueError("Source cohort or baseline differs from frozen settings")
    return prior


def point_metrics(run, labels, has_groups):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    summary = read_json(run / "summary.json")
    with np.load(run / "mixture.npz", allow_pickle=False) as arrays:
        scores, predicted = arrays["scores"], arrays["labels"]
    return {
        "ari": float(adjusted_rand_score(labels, predicted)) if has_groups else None,
        "joint_axis_r2": axis_r2(scores, labels) if has_groups else None,
        "bic_terms": bic_terms(
            read_json(run / "candidates.json"),
            summary["selected_k"],
            len(labels),
            scores.shape[1],
            summary["covariance_type"],
        ),
    }


def _fit_dataset(job):
    import numpy as np

    from super_deli.workflows.calibration_artifacts import fit_pipeline
    from super_deli.workflows.integrative import run_integrative

    root, config, declaration = job
    cohort, i = declaration["cohort"], declaration["repetition"]
    directory = root / cohort / f"dataset_{i:04d}"
    labels = np.asarray(read_json(directory / "truth.json")["labels"])
    dataset = load_config(directory / "raw/dataset.json")
    records = []
    for family in FAMILIES:
        model = study_model(config, cohort, i, family)
        staging = directory / family
        staging.mkdir()
        record = fit_pipeline(
            staging, dataset, directory / "raw", family, model=model, runner=run_integrative
        )
        if record["status"] == "complete":
            summary = read_json(staging / "run/summary.json")
            if summary["unit_ids"] != read_json(directory / "truth.json")["unit_ids"]:
                raise ValueError("Evaluation identities differ from fitted units")
            replicates = read_json(staging / "run/bootstrap_replicates.json")
            record["metrics"] = point_metrics(staging / "run", labels, declaration["has_groups"])
            record["bootstrap"] = {
                "requested": model.n_bootstrap,
                "attempted": len(replicates),
                "successful": sum(r["status"] == "complete" for r in replicates),
                "failed": sum(r["status"] == "failed" for r in replicates),
            }
        else:
            record["bootstrap"] = {
                "requested": model.n_bootstrap,
                "attempted": 0,
                "successful": 0,
                "failed": 0,
            }
        write_json(record, staging / "record.json")
        records.append({"cohort": cohort, "repetition": i, "family": family, **record})
    if all(r["status"] == "complete" for r in records):
        with (
            np.load(directory / "full/run/mixture.npz", allow_pickle=False) as full,
            np.load(directory / "tied/run/mixture.npz", allow_pickle=False) as tied,
        ):
            np.testing.assert_array_equal(full["scores"], tied["scores"])
    return records


def accounting(records):
    return {
        "planned_points": len(records),
        "successful_points": sum(r["status"] == "complete" for r in records),
        "failed_points": sum(r["status"] == "failed" for r in records),
        "planned_bootstrap": sum(r["bootstrap"]["requested"] for r in records),
        "attempted_bootstrap": sum(r["bootstrap"]["attempted"] for r in records),
        "successful_bootstrap": sum(r["bootstrap"]["successful"] for r in records),
        "failed_bootstrap": sum(r["bootstrap"]["failed"] for r in records),
        "unattempted_bootstrap": sum(
            r["bootstrap"]["requested"] - r["bootstrap"]["attempted"] for r in records
        ),
        "significance_fits": 0,
    }


def run_shared_study(config, *, source_artifacts, output_dir, workers=1):
    try:
        for name in ("numpy", "scipy", "sklearn", "snpio"):
            import_module(name)
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Install 'super-deli[genomics,clustering]' for the shared-covariance study"
        ) from exc
    import numpy as np

    config = SharedStudyConfig.from_dict(config.to_dict())
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four independent single-thread workers")
    root, source = Path(output_dir).resolve(), Path(source_artifacts).resolve()
    if root == source or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Study output must be separate from immutable source artifacts")
    if root.exists():
        raise FileExistsError(f"Output already exists: {root}")
    prior = source_check(source, config)
    root.mkdir()
    write_json(config.to_dict(), root / "configuration.json")
    write_json(prior.to_dict(), root / "source_study.json")
    write_json(
        {
            "workers": workers,
            "families": list(FAMILIES),
            "unequal_variance_sd_multipliers": [0.5, 0.8, 1.2, 1.8],
            "source_artifact_id": "integrative-recovery-20261021",
        },
        root / "execution.json",
    )
    declarations = []
    for cohort in COHORTS:
        (root / cohort).mkdir()
        for i in range(config.n_datasets):
            n = config.large_n if cohort == "fresh_256" else config.small_n
            seed = (
                derived_seed(prior.random_seed, "complementary", i, "observed_data")
                if cohort == "archived"
                else study_seed(config, cohort, i)
            )
            directory = root / cohort / f"dataset_{i:04d}"
            if cohort == "archived":
                original = source / "study/archived" / directory.name
                shutil.copytree(
                    original,
                    directory,
                    ignore=shutil.ignore_patterns(
                        "baseline",
                        "pcs_3",
                        "pcs_4",
                        "ecology_0.5",
                        "ecology_2",
                        "ecology_4",
                        "starts_100",
                        "truth_initialized",
                    ),
                )
                shutil.copytree(original / "baseline", directory / "source_baseline")
            else:
                stage_generated(*simulate_shared(cohort, n, seed), directory)
            declarations.append(
                {
                    "cohort": cohort,
                    "repetition": i,
                    "n": n,
                    "seed": seed,
                    "has_groups": cohort not in {"zero_association", "cline"},
                }
            )
    write_json(declarations, root / "datasets.json")
    jobs = [(root, config, row) for row in declarations]
    if workers == 1:
        records = [r for job in jobs for r in _fit_dataset(job)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            records = [r for outcome in executor.map(_fit_dataset, jobs) for r in outcome]
    write_json(records, root / "points.json")
    replay = []
    for i in range(config.n_datasets):
        base = root / "archived" / f"dataset_{i:04d}"
        old, new = (
            read_json(base / "source_baseline/record.json"),
            read_json(base / "full/record.json"),
        )
        if new["status"] == "failed":
            replay.append({"repetition": i, "status": "failed", "error": new["error"]})
            continue
        if old["selected_k"] != new["selected_k"]:
            raise ValueError("Archived full-covariance selection did not replay")
        for a, b in zip(old["candidates"], new["candidates"], strict=True):
            if {k: v for k, v in a.items() if k != "bic"} != {
                k: v for k, v in b.items() if k != "bic"
            }:
                raise ValueError("Archived candidate constraints did not replay")
            np.testing.assert_allclose(a["bic"], b["bic"], rtol=1e-10, atol=1e-7)
        with (
            np.load(base / "source_baseline/run/mixture.npz", allow_pickle=False) as a,
            np.load(base / "full/run/mixture.npz", allow_pickle=False) as b,
        ):
            np.testing.assert_array_equal(a["labels"], b["labels"])
            np.testing.assert_allclose(a["scores"], b["scores"], atol=1e-11, rtol=1e-9)
        replay.append({"repetition": i, "status": "reproduced"})
    write_json(replay, root / "replay.json")
    result = {
        "schema_version": 1,
        "datasets": len(declarations),
        "reproduced_archived": sum(r["status"] == "reproduced" for r in replay),
        **accounting(records),
    }
    write_json(result, root / "summary.json")
    write_json(inventory(root), root / "files.json")
    return result
