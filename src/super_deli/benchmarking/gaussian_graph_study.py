"""Frozen, gated benchmark lifecycle with bounded submission and local evidence."""

import platform
import shutil
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path

from super_deli import __version__
from super_deli.benchmarking.gaussian_graph_config import SETTINGS, StudyConfig, counts, tasks
from super_deli.benchmarking.neural_study import retained_bytes, source_hashes
from super_deli.benchmarking.unknown_k_data import arm_declarations
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def declaration(root, config):
    from super_deli.models.spectral_graph import helper_fingerprint

    return {
        "configuration": config.to_dict(),
        "settings": SETTINGS,
        "baselines": arm_declarations(),
        "counts": {p: counts(config, p) for p in ("preflight", "main")},
        "source_sha256": source_hashes(),
        "plan_sha256": fingerprint_file(root / "approved-plan.md").sha256,
        "qr_helper_sha256": helper_fingerprint(),
        "python": sys.version,
        "platform": platform.platform(),
        "package_version": __version__,
        "software": {p: version(p) for p in ("numpy", "scipy", "scikit-learn", "threadpoolctl")},
    }


def freeze(root, config):
    root = Path(root)
    config = StudyConfig.from_dict(config.to_dict())
    result = declaration(root, config)
    if not config.fixture:
        assert result["counts"]["main"]["evaluations"] == 97200
        assert result["counts"]["preflight"]["evaluations"] == 10800
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
    from super_deli.benchmarking.gaussian_graph_run import run_dataset

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
    frozen, config = check_frozen(root)
    if type(staging_bytes) is not int or staging_bytes < 0:
        raise ValueError("Measure staging bytes")
    audit = read_json(root / "preflight-audit.json")
    execution = read_json(root / "preflight-execution.json")
    if audit["execution_sha256"] != fingerprint_file(root / "preflight-execution.json").sha256:
        raise ValueError("Execution ledger changed after audit")
    ratio = frozen["counts"]["main"]["evaluations"] / frozen["counts"]["preflight"]["evaluations"]
    worst = {
        n: max(t["seconds"] for d in execution["datasets"] if d["n"] == n for t in d["tables"])
        for n in {t["n"] for t in tasks(config, "main")}
    }
    # Serial worst-case table times, plus at least 0.1 s per possible assignment;
    # never rely on parallel speedups or cheap K=1/withheld assignments to pass.
    timing = sum(
        (2 + len(t["replay_conditions"])) * (worst[t["n"]] + 0.4) for t in tasks(config, "main")
    )
    seconds = 1.25 * (timing + audit["seconds"] * ratio)
    projected = int(
        1.25 * retained_bytes(root / "preflight") * ratio
        + retained_bytes(root)
        + 2 * staging_bytes
        + 128 * 1024**2
    )
    free = shutil.disk_usage(root).free
    result = {
        "passed": audit["passed"]
        and seconds < config.max_projected_seconds
        and projected <= config.artifact_cap_bytes
        and free >= config.min_free_bytes,
        "projected_bytes": projected,
        "projected_main_seconds_conservative": seconds,
        "staging_bytes": staging_bytes,
        "staging_copies_counted": 2,
        "free_bytes": free,
        "audit_sha256": fingerprint_file(root / "preflight-audit.json").sha256,
        "execution_sha256": fingerprint_file(root / "preflight-execution.json").sha256,
        "freeze_sha256": fingerprint_file(root / "frozen.json").sha256,
    }
    write_json(result, root / "storage-runtime-gate.json")
    return result
