"""Frozen observed-GMM diagnostic orchestration; no production model promotion."""

import json
import math
import shutil
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path

from super_deli.benchmarking.observed_recovery import (
    digest,
    extend_search,
    load_arrays,
    oracle_start,
)
from super_deli.benchmarking.observed_study import NAMES, masks, sample, save_fit, source_identity
from super_deli.io.config import read_json, write_json

CASES = ("spherical", "correlated", "four_groups")
MASKS = ("mcar30", "mar")
ARMS = ("observed", "reference")
ARCHIVED_FAILURES = (
    ("four_groups_n64_05", "mar_observed"),
    ("four_groups_n64_08", "mar_observed"),
    ("spherical_n64_04", "pattern_observed"),
)
SETTINGS = dict(
    schema_version=1,
    cases=list(CASES),
    masks=list(MASKS),
    arms=list(ARMS),
    sample_sizes=[64, 256],
    replicates=10,
    evaluation_units=2048,
    data_seed=81403,
    mask_seed=81409,
    evaluation_seed=81429,
    model_seed=80423,
    total_iteration_cap=10000,
    main_baseline_searches=252,
    preflight_baseline_searches=30,
    main_oracle_starts=160,
    preflight_oracle_starts=8,
    max_fresh_continuations=9600,
    archived_continuations=30,
    max_workers=4,
    artifact_cap_bytes=2 * 1024**3,
    min_free_bytes=8 * 1024**3,
    allowance=1.25,
    reserve_bytes=128 * 1024**2,
)


def tasks(phase):
    if phase not in ("preflight", "main"):
        raise ValueError("Unknown diagnostic phase")
    return [
        dict(
            key=f"{case}_n{n}_{i:02}",
            case=case,
            n=n,
            replicate=i,
            phase=phase,
            namespace=[2 if phase == "preflight" else 3, c, n, i],
        )
        for c, case in enumerate(CASES)
        for n in (64, 256)
        for i in range(1 if phase == "preflight" else 10)
    ]


def make_inputs(task):
    x, truth = sample(task["case"], task["n"], [81403] + task["namespace"])
    ev, ev_truth = sample(task["case"], 2048, [81429] + task["namespace"] + [0])
    missing = masks(x, [81409] + task["namespace"])
    evaluation_missing = masks(ev, [81429] + task["namespace"] + [1])
    return dict(
        complete=x,
        truth=truth,
        evaluation=ev,
        evaluation_truth=ev_truth,
        **{f"mask_{name}": missing[name] for name in MASKS},
        **{f"eval_mask_{name}": evaluation_missing[name] for name in MASKS},
    )


def directory_manifest(root):
    return {
        p.relative_to(root).as_posix(): dict(bytes=p.stat().st_size, sha256=digest(p))
        for p in sorted(Path(root).rglob("*"))
        if p.is_file() and not p.is_symlink()
    }


def freeze(root, archive):
    from super_deli import __version__
    from super_deli.models.observed_config import ObservedGMMConfig

    root, archive = Path(root).resolve(), Path(archive).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.glob("*flight")) or (root / "frozen-before-generation.json").exists():
        raise ValueError("Diagnostic freeze must precede generation")
    if read_json(archive / "final-verification.json")["status"] != "complete":
        raise ValueError("A completed checksum-bound archival study is required")
    old_source = read_json(archive / "execution-freeze.json")["source_sha256"]
    current_source = source_identity()
    if any(
        current_source.get(name) != value
        for name, value in old_source.items()
        if name != "__init__.py"
    ):
        raise ValueError("Existing scientific sources changed since the completed study")
    software = {
        name: version(name)
        for name in ("super-deli", "numpy", "scipy", "scikit-learn", "threadpoolctl")
    }
    if software["super-deli"] != __version__:
        raise ValueError("Imported source and installed metadata differ")
    result = dict(
        status="frozen_before_generation",
        settings=SETTINGS,
        model=ObservedGMMConfig(NAMES).to_dict(),
        source_sha256=source_identity(),
        software=software,
        archive=str(archive),
        archive_manifest=directory_manifest(archive),
        production_modules_unchanged=True,
        pvalues=False,
        empirical_fits=False,
        bootstrap=False,
        author_contact=False,
    )
    write_json(result, root / "frozen-before-generation.json")
    return result


def verify_freeze(root):
    root = Path(root)
    frozen = read_json(
        root
        / (
            "execution-freeze.json"
            if (root / "execution-freeze.json").exists()
            else "frozen-before-generation.json"
        )
    )
    if frozen["settings"] != SETTINGS or frozen["source_sha256"] != source_identity():
        raise ValueError("Frozen diagnostic settings/source changed")
    if any(version(name) != value for name, value in frozen["software"].items()):
        raise ValueError("Frozen scientific software versions changed")
    return frozen


def storage_check(root):
    root = Path(root)
    if shutil.disk_usage(root).free < SETTINGS["min_free_bytes"]:
        raise ValueError("Diagnostic requires at least 8 GiB free")
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    if size > SETTINGS["artifact_cap_bytes"]:
        raise ValueError("Diagnostic local-artifact cap exceeded")
    return size


def observation(arguments):
    import numpy as np

    from super_deli.models.observed_config import ObservedGMMConfig
    from super_deli.models.observed_gmm import normalization

    task, root = arguments
    storage_check(root)
    directory = Path(root) / task["phase"] / task["key"]
    directory.mkdir()
    data = make_inputs(task)
    np.savez_compressed(directory / "inputs.npz", **data)
    write_json(task, directory / "task.json")
    searches = continuations = oracles = 0
    for name in MASKS:
        masked = np.where(data[f"mask_{name}"], np.nan, data["complete"])
        norm = normalization(masked)
        for arm in ARMS:
            values = masked if arm == "observed" else data["complete"]
            condition = directory / f"{name}_{arm}"
            condition.mkdir()
            metadata = dict(mask=name, arm=arm, task=task, original_n=task["n"])
            args = (
                values,
                norm,
                ObservedGMMConfig(NAMES),
                condition / "baseline",
                metadata,
                np.arange(task["n"]),
                data["evaluation"],
                data[f"eval_mask_{name}"],
                data["truth"],
            )
            save_fit(*args)
            searches += 1
            if (condition / "baseline" / "fit.npz").exists():
                result = extend_search(
                    condition / "baseline",
                    condition / "extended",
                    data["evaluation"],
                    data[f"eval_mask_{name}"],
                    data["truth"],
                )
                continuations += result["continuation_attempts"]
                if task["case"] == "four_groups":
                    oracle_start(condition / "baseline", condition / "oracle")
                    oracles += 1
            if task["phase"] == "preflight" and name == "mcar30" and arm == "observed":
                replay_args = list(args)
                replay_args[3] = condition / "replay"
                save_fit(*replay_args)
                searches += 1
    storage_check(root)
    return dict(key=task["key"], searches=searches, continuations=continuations, oracles=oracles)


def archived_job(arguments):
    """One bounded archive action, writing exclusively into the new study root."""
    import numpy as np

    from super_deli.models.observed_config import ObservedGMMConfig
    from super_deli.models.observed_gmm import Normalization

    root, archive, action, key, condition = arguments
    root, archive = Path(root), Path(archive)
    storage_check(root)
    source = archive / "main" / key / condition
    data = load_arrays(archive / "main" / key / "inputs.npz")
    summary = read_json(source / "summary.json")
    if action == "continuation":
        output = root / "archived-continuations" / f"{key}__{condition}"
        result = extend_search(
            source,
            output,
            data["evaluation"],
            data[f"eval_mask_{summary['mask']}"],
            data["truth"],
            components=(4,),
        )
        return dict(
            action=action, key=key, condition=condition, attempts=result["continuation_attempts"]
        )
    if action == "oracle":
        output = root / "archived-oracles" / f"{key}__{condition}"
        oracle_start(source, output)
    elif action == "replay":
        a = load_arrays(source / "fit.npz")
        mask = data[f"mask_{summary['mask']}"]
        values = np.where(mask, np.nan, data["complete"])
        output = root / "archived-replays" / f"{key}__{condition}"
        metadata = {name: summary[name] for name in ("mask", "arm", "task", "original_n")}
        save_fit(
            values,
            Normalization(a["normalization_means"], a["scales"]),
            ObservedGMMConfig.from_dict(summary["config"]),
            output,
            metadata,
            np.arange(len(values)),
            data["evaluation"],
            data[f"eval_mask_{summary['mask']}"],
            data["truth"],
        )
    else:
        raise ValueError("Unknown archived action")
    return dict(action=action, key=key, condition=condition)


def run_phase(root, phase, workers=4):
    root = Path(root)
    frozen = verify_freeze(root)
    if not 1 <= workers <= 4:
        raise ValueError("Use at most four one-thread workers")
    if phase == "main" and read_json(root / "storage-gate.json")["status"] != "passed":
        raise ValueError("Main run requires the passed independent preflight/storage gate")
    (root / phase).mkdir()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        outcomes = list(executor.map(observation, [(t, root) for t in tasks(phase)]))
        archived = []
        if phase == "main":
            for action in ("continuations", "oracles", "replays"):
                (root / f"archived-{action}").mkdir()
            jobs = [
                (root, frozen["archive"], "continuation", key, condition)
                for key, condition in ARCHIVED_FAILURES
            ]
            jobs += [
                (root, frozen["archive"], "oracle", f"four_groups_n{n}_{i:02}", f"{mask}_{arm}")
                for n in (64, 256)
                for i in range(10)
                for mask in MASKS
                for arm in ARMS
            ]
            jobs += [
                (root, frozen["archive"], "replay", f"{case}_n{n}_00", f"{mask}_observed")
                for case in CASES
                for n in (64, 256)
                for mask in MASKS
            ]
            archived = list(executor.map(archived_job, jobs))
    result = dict(status="completed", phase=phase, observations=outcomes, archived=archived)
    write_json(result, root / f"{phase}-execution.json")
    return result


def gate(root):
    root = Path(root)
    verify_freeze(root)
    audit = read_json(root / "preflight-audit.json")
    if audit["status"] != "passed" or audit["searches"] != 30 or audit["oracles"] != 8:
        raise ValueError("Complete independently audited preflight required")
    if read_json(root / "archive-audit.json")["status"] != "passed":
        raise ValueError("Complete archival audit required")
    measured = sum(p.stat().st_size for p in (root / "preflight").rglob("*") if p.is_file())
    oracle_bytes = sum(
        p.stat().st_size for p in (root / "preflight").glob("*/*/oracle/*") if p.is_file()
    )
    retained = storage_check(root)
    # Project measured full searches x10, another 80 archive oracle starts, then
    # conservatively allow every feasible fresh start and all 30 archive paths
    # to store 8,000 additional float64 trace entries plus 16 KiB of metadata.
    continuation_ceiling = (9600 + 30) * (8000 * 8 + 16 * 1024)
    projected = math.ceil(
        retained + 1.25 * (10 * measured + 10 * oracle_bytes + continuation_ceiling) + 128 * 1024**2
    )
    free = shutil.disk_usage(root).free
    result = dict(
        status="passed" if projected <= 2 * 1024**3 and free >= 8 * 1024**3 else "failed",
        measured_preflight_bytes=measured,
        measured_oracle_bytes=oracle_bytes,
        continuation_worst_case_bytes=continuation_ceiling,
        retained_bytes=retained,
        projected_bytes=projected,
        cap_bytes=2 * 1024**3,
        free_bytes=free,
        allowance=1.25,
        reserve_bytes=128 * 1024**2,
    )
    write_json(result, root / "storage-gate.json")
    return result


def main():
    import argparse
    import os

    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=[
            "freeze",
            "archive-audit",
            "preflight",
            "audit-preflight",
            "gate",
            "main",
            "audit-main",
            "report",
        ],
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--archive", help="Completed local observed-data study; freeze only")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.phase == "freeze":
        if not args.archive:
            parser.error("freeze requires --archive")
        result = freeze(args.output_dir, args.archive)
        print(json.dumps({key: result[key] for key in ("status", "settings")}))
        return
    verify_freeze(args.output_dir)
    if args.phase in ("preflight", "main"):
        result = run_phase(args.output_dir, args.phase, args.workers)
    elif args.phase == "gate":
        result = gate(args.output_dir)
    else:
        from super_deli.benchmarking.observed_recovery_audit import (
            archive_audit,
            audit_phase,
            report,
        )

        if args.phase == "archive-audit":
            result = archive_audit(args.output_dir, args.workers)
        elif args.phase.startswith("audit-"):
            result = audit_phase(args.output_dir, args.phase[6:], args.workers)
        else:
            result = report(args.output_dir)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
