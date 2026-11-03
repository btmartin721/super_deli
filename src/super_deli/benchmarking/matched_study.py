"""Frozen, gated benchmark lifecycle with bounded submission and local evidence."""

import json
import platform
import shutil
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path

from super_deli import __version__
from super_deli.benchmarking.matched_config import ARMS, SETTINGS, StudyConfig, counts, tasks
from super_deli.benchmarking.matched_data import configuration
from super_deli.benchmarking.neural_study import retained_bytes, source_hashes
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def declaration(root, config):
    from super_deli.models.neural_config import TRAINING

    return {
        "configuration": config.to_dict(),
        "settings": SETTINGS,
        "arms": json.loads(json.dumps({arm: configuration(arm, 0).to_dict() for arm in ARMS})),
        "training": TRAINING,
        "counts": {p: counts(config, p) for p in ("preflight", "main")},
        "source_sha256": source_hashes(),
        "plan_sha256": fingerprint_file(root / "approved-plan.md").sha256,
        "python": sys.version,
        "platform": platform.platform(),
        "package_version": __version__,
        "software": {
            p: version(p) for p in ("numpy", "scipy", "scikit-learn", "threadpoolctl", "torch")
        },
    }


def freeze(root, config):
    root = Path(root)
    config = StudyConfig.from_dict(config.to_dict())
    result = declaration(root, config)
    if not config.fixture:
        assert result["counts"]["main"]["workflows"] == 2708
        assert result["counts"]["preflight"]["workflows"] == 472
    write_json(result, root / "frozen.json")
    package = Path(__file__).resolve().parents[1]
    with tarfile.open(root / "frozen-sources.tar.gz", "x:gz") as f:
        for p in sorted(package.rglob("*.py")):
            f.add(p, arcname=str(p.relative_to(package)))
    return {"passed": True, "counts": result["counts"], "fixture": config.fixture}


def check_frozen(root):
    root = Path(root)
    frozen = read_json(root / "frozen.json")
    config = StudyConfig.from_dict(frozen["configuration"])
    if declaration(root, config) != frozen:
        raise ValueError("Frozen source, configuration, plan or numerical runtime differs")
    return frozen, config


def inventory(folder):
    return {
        str(p.relative_to(folder)): fingerprint_file(p).sha256
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def guard(root, config, staging_bytes=0):
    if retained_bytes(root) + staging_bytes > config.artifact_cap_bytes:
        raise ValueError("Artifact cap exceeded")
    if shutil.disk_usage(root).free < config.min_free_bytes:
        raise ValueError("Free-space gate")


def run_phase(root, phase, workers):
    from super_deli.benchmarking.matched_run import run_dataset

    root = Path(root)
    frozen, config = check_frozen(root)
    if type(workers) is not int or not 1 <= workers <= config.max_workers:
        raise ValueError("Worker count exceeds frozen limit")
    staging = 0
    if phase == "main":
        gate = read_json(root / "storage-runtime-gate.json")
        if not gate["passed"] or any(
            gate[key] != fingerprint_file(root / name).sha256
            for key, name in (
                ("freeze_sha256", "frozen.json"),
                ("audit_sha256", "preflight-audit.json"),
                ("execution_sha256", "preflight-execution.json"),
            )
        ):
            raise ValueError("Passed matching audited preflight gate required")
        staging = 2 * gate["staging_bytes"]
    guard(root, config, staging)
    grid = tasks(config, phase)
    out = root / phase
    out.mkdir()
    start = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for offset in range(0, len(grid), workers):
            for result in pool.map(
                run_dataset, [(str(out), t) for t in grid[offset : offset + workers]]
            ):
                results.append(result)
                if len(results) % 10 == 0 or len(results) == len(grid):
                    print(
                        {"phase": phase, "observations": len(results), "total": len(grid)},
                        flush=True,
                    )
                guard(root, config, staging)
                if time.monotonic() - start > config.max_projected_seconds:
                    raise ValueError("Runtime limit exceeded")
    write_json(inventory(out), root / f"{phase}-inventory.json")
    result = {
        "phase": phase,
        "seconds": time.monotonic() - start,
        "datasets": results,
        "workers": workers,
        **frozen["counts"][phase],
    }
    write_json(result, root / f"{phase}-execution.json")
    return {k: v for k, v in result.items() if k != "datasets"}


def gate(root, staging_bytes):
    root = Path(root)
    f, config = check_frozen(root)
    if type(staging_bytes) is not int or staging_bytes < 0:
        raise ValueError("Measure staging bytes")
    audit = read_json(root / "preflight-audit.json")
    execution = read_json(root / "preflight-execution.json")
    if audit["execution_sha256"] != fingerprint_file(root / "preflight-execution.json").sha256:
        raise ValueError("Execution ledger changed after audit")
    timing = {}
    overhead = {}
    point_bytes = {}
    native_bytes = {}
    source_bytes = {}
    for data in execution["datasets"]:
        folder = root / "preflight" / data["key"]
        n = data["n"]
        source_bytes[n] = max(
            source_bytes.get(n, 0),
            sum(retained_bytes(folder / p) for p in ("raw", "canonical"))
            + (folder / "generating-reference.npz").stat().st_size,
        )
        for row in data["records"]:
            arm = row["arm"]
            out = folder / (arm + ("_replay" if row["kind"] == "replay" else ""))
            events = read_json(out / "fit-times.json") if (out / "fit-times.json").exists() else []
            if arm.endswith("gmm"):
                events = [
                    {
                        "fit": 0,
                        "status": row["status"] == "available" and "complete" or "failed",
                        "seconds": row["seconds"],
                    }
                ]
            for event in events:
                if event["status"] != "complete":
                    continue
                role = "point" if event["fit"] == 0 else "refit"
                timing[n, arm, role] = max(timing.get((n, arm, role), 0), event["seconds"])
            overhead[n, arm] = max(
                overhead.get((n, arm), 0), row["seconds"] - sum(e["seconds"] for e in events), 0
            )
            size = retained_bytes(out)
            target = native_bytes if row["draws_requested"] else point_bytes
            target[n, arm] = max(target.get((n, arm), 0), size)
    missing = []
    seconds = 0.0
    projected = 0
    for task in tasks(config, "main"):
        n = task["n"]
        projected += source_bytes[n] + 4096
        for arm in ARMS:
            roles = [("point", 1 + int(task["replay"]))]
            if task["draws"] and not arm.endswith("gmm"):
                roles.append(("refit", task["draws"]))
            for role, count in roles:
                if (n, arm, role) not in timing:
                    missing.append([n, arm, role])
                seconds += count * timing.get((n, arm, role), 0)
            seconds += (1 + int(task["replay"])) * overhead[n, arm]
            if (n, arm) not in point_bytes:
                missing.append([n, arm, "point_bytes"])
            projected += (1 + int(task["replay"])) * point_bytes.get((n, arm), 0)
            if task["draws"] and not arm.endswith("gmm"):
                if (n, arm) not in native_bytes:
                    missing.append([n, arm, "native_bytes"])
                projected += native_bytes.get((n, arm), 0)
    # Largest workload ratio bounds fixed per-observation and all per-fit audit
    # work, including every NumPy optimizer replay. No parallel speed credit.
    ratio = max(
        f["counts"]["main"][k] / f["counts"]["preflight"][k]
        for k in ("observations", "point_fits", "subsample_refits", "point_replays", "neural_fits")
    )
    seconds = 1.25 * (seconds + audit["seconds"] * ratio)
    projected = int(1.25 * projected + retained_bytes(root) + 2 * staging_bytes + 128 * 1024**2)
    free = shutil.disk_usage(root).free
    result = {
        "passed": audit["passed"]
        and not missing
        and seconds < config.max_projected_seconds
        and projected <= config.artifact_cap_bytes
        and free >= config.min_free_bytes,
        "projected_bytes": projected,
        "projected_main_seconds_conservative": seconds,
        "missing_timing_storage_strata": missing,
        "staging_bytes": staging_bytes,
        "staging_copies_counted": 2,
        "free_bytes": free,
        "audit_sha256": fingerprint_file(root / "preflight-audit.json").sha256,
        "execution_sha256": fingerprint_file(root / "preflight-execution.json").sha256,
        "freeze_sha256": fingerprint_file(root / "frozen.json").sha256,
    }
    write_json(result, root / "storage-runtime-gate.json")
    return result


def dispatch_study(args):
    config = (
        StudyConfig.from_dict(read_json(args.study_config)) if args.study_config else StudyConfig()
    )
    root = args.output_dir.resolve()
    if args.action == "freeze":
        return freeze(root, config)
    _, frozen = check_frozen(root)
    if args.study_config and config != frozen:
        raise ValueError("Configuration differs from freeze")
    if args.action in ("main", "preflight"):
        return run_phase(root, args.action, args.workers)
    if args.action == "gate":
        return gate(root, args.staging_bytes)
    if args.action.startswith("audit-"):
        from super_deli.benchmarking.matched_audit import audit_phase

        return audit_phase(root, args.action.removeprefix("audit-"))
    from super_deli.benchmarking.matched_report import report

    return report(root)
