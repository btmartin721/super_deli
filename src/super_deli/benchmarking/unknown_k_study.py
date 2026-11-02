"""Frozen fresh point study with audited storage/runtime gates."""

import platform
import shutil
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path

from super_deli.benchmarking.neural_study import retained_bytes, source_hashes
from super_deli.benchmarking.unknown_k_config import ARMS, SETTINGS, StudyConfig, counts, tasks
from super_deli.benchmarking.unknown_k_data import arm_declarations
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def freeze(root, config):
    from super_deli.models.spectral_graph import helper_fingerprint

    root = Path(root)
    config = StudyConfig.from_dict(config.to_dict())
    result = {
        "configuration": config.to_dict(),
        "settings": SETTINGS,
        "arms": arm_declarations(),
        "counts": {p: counts(config, p) for p in ("preflight", "main")},
        "source_sha256": source_hashes(),
        "plan_sha256": fingerprint_file(root / "approved-plan.md").sha256,
        "qr_helper_sha256": helper_fingerprint(),
        "python": sys.version,
        "platform": platform.platform(),
        "software": {
            p: version(p) for p in ("super-deli", "numpy", "scipy", "scikit-learn", "threadpoolctl")
        },
    }
    if not config.fixture:
        assert result["counts"]["main"]["evaluations"] == 2592
        assert result["counts"]["preflight"]["evaluations"] == 154
    write_json(result, root / "frozen.json")
    package = Path(__file__).resolve().parents[1]
    with tarfile.open(root / "frozen-sources.tar.gz", "x:gz") as f:
        for p in sorted(package.rglob("*.py")):
            f.add(p, arcname=str(p.relative_to(package)))
    return {"passed": True, "counts": result["counts"], "fixture": config.fixture}


def check_frozen(root):
    from super_deli.models.spectral_graph import helper_fingerprint

    f = read_json(root / "frozen.json")
    config = StudyConfig.from_dict(f["configuration"])
    if (
        f["settings"] != SETTINGS
        or f["source_sha256"] != source_hashes()
        or f["arms"] != arm_declarations()
        or f["counts"] != {p: counts(config, p) for p in ("preflight", "main")}
        or f["plan_sha256"] != fingerprint_file(root / "approved-plan.md").sha256
        or f["qr_helper_sha256"] != helper_fingerprint()
        or f["python"] != sys.version
        or f["platform"] != platform.platform()
        or f["software"] != {p: version(p) for p in f["software"]}
    ):
        raise ValueError("Frozen source, configuration, plan or numerical runtime differs")
    return f, config


def run_phase(root, phase, workers):
    from super_deli.benchmarking.unknown_k_run import probe, run_dataset

    frozen, config = check_frozen(root)
    if type(workers) is not int or not 1 <= workers <= config.max_workers:
        raise ValueError("Worker count exceeds frozen limit")
    if shutil.disk_usage(root).free < config.min_free_bytes:
        raise ValueError("Free-space gate")
    if retained_bytes(root) > config.artifact_cap_bytes:
        raise ValueError("Artifact cap exceeded before execution")
    if phase == "main":
        g = read_json(root / "storage-runtime-gate.json")
        if (
            not g["passed"]
            or g["freeze_sha256"] != fingerprint_file(root / "frozen.json").sha256
            or g["audit_sha256"] != fingerprint_file(root / "preflight-audit.json").sha256
            or g["execution_sha256"] != fingerprint_file(root / "preflight-execution.json").sha256
        ):
            raise ValueError("Passed matching audited preflight gate required")
    grid = tasks(config, phase)
    out = root / phase
    out.mkdir()
    start = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        # Submit only one bounded batch; a failed resource guard cannot leave
        # the entire scientific grid queued behind it.
        for offset in range(0, len(grid), workers):
            batch = grid[offset : offset + workers]
            for result in pool.map(run_dataset, [(str(out), t) for t in batch]):
                results.append(result)
                if (len(results) - 1) % 20 == 0 or len(results) == len(grid):
                    print(
                        {"phase": phase, "observations": len(results), "total": len(grid)},
                        flush=True,
                    )
                if retained_bytes(root) > config.artifact_cap_bytes:
                    raise ValueError("Artifact cap exceeded")
                if shutil.disk_usage(root).free < config.min_free_bytes:
                    raise ValueError("Free space exhausted")
                if phase == "main" and time.monotonic() - start > config.max_projected_seconds:
                    raise ValueError("Main runtime limit exceeded")
    probes = []
    if phase == "preflight":
        for n in (32, 48) if config.fixture else (64, 256):
            for k in range(2, 7):
                probes.append(probe(out / "probes" / f"n{n}_k{k}", n, k))
    inventory = {
        str(p.relative_to(out)): fingerprint_file(p).sha256
        for p in sorted(out.rglob("*"))
        if p.is_file()
    }
    write_json(inventory, root / f"{phase}-inventory.json")
    result = {
        "phase": phase,
        "seconds": time.monotonic() - start,
        "datasets": results,
        "probe_fits": probes,
        "workers": workers,
        **frozen["counts"][phase],
    }
    write_json(result, root / f"{phase}-execution.json")
    return {k: result[k] for k in ("phase", "seconds", "evaluations", "workflows", "observations")}


def gate(root, staging_bytes):
    f, config = check_frozen(root)
    if type(staging_bytes) is not int or staging_bytes < 0:
        raise ValueError("Measure staging bytes")
    audit = read_json(root / "preflight-audit.json")
    execution = read_json(root / "preflight-execution.json")
    if audit["execution_sha256"] != fingerprint_file(root / "preflight-execution.json").sha256:
        raise ValueError("Execution ledger changed after audit")
    by_size = {}
    missing = []
    for n in (32, 48) if config.fixture else (64, 256):
        probes = [
            p
            for p in execution["probe_fits"]
            if p["n"] == n and p["status"] == "complete" and p.get("reason") is None
        ]
        if {p["target"] for p in probes} != set(range(2, 7)):
            missing.append([n, "embedding_probes"])
        worst_probe = max((p["seconds"] for p in probes), default=0)
        for arm in ARMS:
            runs = [
                r
                for d in execution["datasets"]
                if d["n"] == n
                for r in d["records"]
                if r["arm"] == arm
            ]
            eligible = [r["seconds"] for r in runs if r["status"] == "available"]
            if arm in {"gmm", "density"} and not eligible:
                missing.append([n, arm])
            measured = max((r["seconds"] for r in runs), default=0)
            by_size[n, arm] = measured + (worst_probe if arm.startswith("spectral") else 0)
    seconds = sum(
        (1 + t["replay"]) * 2 * sum(by_size[t["n"], a] for a in ARMS) for t in tasks(config, "main")
    )
    ratio = f["counts"]["main"]["evaluations"] / f["counts"]["preflight"]["evaluations"]
    projected_seconds = 1.25 * (seconds + audit["seconds"] * ratio)
    projected = int(
        retained_bytes(root / "preflight") * ratio * 1.25
        + retained_bytes(root)
        + 2 * staging_bytes
        + 128 * 1024**2
    )
    free = shutil.disk_usage(root).free
    result = {
        "passed": audit["passed"]
        and not missing
        and projected_seconds < config.max_projected_seconds
        and projected <= config.artifact_cap_bytes
        and free >= config.min_free_bytes,
        "projected_bytes": projected,
        "projected_main_seconds_conservative": projected_seconds,
        "staging_bytes": staging_bytes,
        "staging_copies_counted": 2,
        "free_bytes": free,
        "missing_timing_strata": missing,
        "audit_sha256": fingerprint_file(root / "preflight-audit.json").sha256,
        "execution_sha256": fingerprint_file(root / "preflight-execution.json").sha256,
        "freeze_sha256": fingerprint_file(root / "frozen.json").sha256,
    }
    write_json(result, root / "storage-runtime-gate.json")
    return result
