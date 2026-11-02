"""Immutable archival graph protocol, execution ledger and audited resource gate."""

import json
import platform
import shutil
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path

from super_deli.benchmarking.neural_study import retained_bytes, source_hashes
from super_deli.benchmarking.spectral_config import (
    SETTINGS,
    SpectralConfig,
    density_jobs,
    graph_jobs,
)
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def freeze(root, config):
    from super_deli.benchmarking.spectral_fit import helper_fingerprint

    config = SpectralConfig.from_dict(config.to_dict())
    binding = read_json(root / "binding.json")
    if config.fixture != (binding["kind"] == "fixture"):
        raise ValueError("Fixture/scientific source kind differs")
    counts = {
        phase: 2 * len(graph_jobs(binding, phase)) + len(density_jobs(binding, phase))
        for phase in ("preflight", "main")
    }
    if not config.fixture and counts != {"preflight": 504, "main": 4628}:
        raise ValueError("Approved evaluation counts differ")
    result = {
        "configuration": config.to_dict(),
        "settings": SETTINGS,
        "counts": counts,
        "source_sha256": source_hashes(),
        "binding_sha256": fingerprint_file(root / "binding.json").sha256,
        "plan_sha256": fingerprint_file(root / "approved-plan.md").sha256,
        "qr_helper_sha256": helper_fingerprint(),
        "python": sys.version,
        "platform": platform.platform(),
        "software": {
            p: version(p) for p in ("super-deli", "numpy", "scipy", "scikit-learn", "threadpoolctl")
        },
    }
    write_json(result, root / "frozen.json")
    package = Path(__file__).resolve().parents[1]
    with tarfile.open(root / "frozen-sources.tar.gz", "x:gz") as bundle:
        for p in sorted(package.rglob("*.py")):
            bundle.add(p, arcname=str(p.relative_to(package)))
    return result


def check_frozen(root):
    from super_deli.benchmarking.spectral_fit import helper_fingerprint

    f = read_json(root / "frozen.json")
    if f["source_sha256"] != source_hashes() or f["settings"] != SETTINGS:
        raise ValueError("Frozen scientific source/settings differ")
    for name, key in (("binding.json", "binding_sha256"), ("approved-plan.md", "plan_sha256")):
        if fingerprint_file(root / name).sha256 != f[key]:
            raise ValueError("Frozen binding/plan differs")
    if (
        f["software"] != {p: version(p) for p in f["software"]}
        or f["python"] != sys.version
        or f["platform"] != platform.platform()
        or f["qr_helper_sha256"] != helper_fingerprint()
    ):
        raise ValueError("Frozen numerical runtime/helper differs")
    return f, SpectralConfig.from_dict(f["configuration"])


def job_path(root, phase, job, density=False):
    suffix = job["kind"] if density else f"k{job['neighbors']}_{job['kind']}"
    return (
        root
        / phase
        / ("density" if density else "graphs")
        / (job["key"] + "__" + job["space"] + "__" + suffix)
    )


def run_graph(args):
    import numpy as np

    from super_deli.benchmarking.spectral_archive import load_coordinates
    from super_deli.benchmarking.spectral_fit import fit_graph

    root, phase, job, entry, config = args
    root = Path(root)
    out = job_path(root, phase, job)
    out.mkdir()
    x = load_coordinates(root, entry)[job["space"]]
    start = time.monotonic()
    try:
        record, data = fit_graph(
            x, entry["unit_ids"], job["neighbors"], max_working_bytes=config.max_working_bytes
        )
    except Exception as exc:
        data = {}
        record = {
            "n": entry["n"],
            "error": str(exc),
            "partitions": [
                {
                    "target_k": k,
                    "status": "failed",
                    "reason": "graph_failure",
                    "solver_called": False,
                }
                for k in (2, 4)
            ],
            "seconds": time.monotonic() - start,
        }
    np.savez_compressed(out / "arrays.npz", **data)
    record["job"] = job
    write_json(record, out / "result.json")
    return {
        "job": job,
        "n": entry["n"],
        "seconds": record["seconds"],
        "partitions": record["partitions"],
    }


def run_density(args):
    import numpy as np
    from threadpoolctl import threadpool_limits

    from super_deli.benchmarking.spectral_archive import load_coordinates
    from super_deli.models.density_scores import fit_density_scores
    from super_deli.models.hdbscan_config import HDBSCANConfig
    from super_deli.models.mixture import FIT_LOCK

    root, phase, job, entry, config = args
    root = Path(root)
    out = job_path(root, phase, job, True)
    out.mkdir()
    a = load_coordinates(root, entry)
    x = a[job["space"]]
    start = time.monotonic()
    data = {}
    try:
        with FIT_LOCK, threadpool_limits(limits=1):
            labels, strength = fit_density_scores(x, HDBSCANConfig(True))
        data = {"scores": x, "labels": labels, "strength": strength}
        record = {"status": "complete", "selected_k": len(set(labels) - {-1})}
        if job["kind"] == "archive-replay":
            np.testing.assert_array_equal(labels, a[job["space"] + "_labels"])
            np.testing.assert_array_equal(strength, a[job["space"] + "_strength"])
            record["exact_archive_replay"] = True
    except Exception as exc:
        record = {"status": "failed", "error": str(exc)}
    record.update(job=job, n=entry["n"], seconds=time.monotonic() - start)
    np.savez_compressed(out / "arrays.npz", **data)
    write_json(record, out / "result.json")
    return record


def run_phase(root, phase, workers=1):
    frozen, config = check_frozen(root)
    if (
        phase not in ("preflight", "main")
        or type(workers) is not int
        or not 1 <= workers <= config.max_workers
    ):
        raise ValueError("Unsupported phase/worker count")
    if shutil.disk_usage(root).free < config.min_free_bytes:
        raise ValueError("At least the frozen free-space minimum is required")
    if phase == "main":
        g = read_json(root / "storage-runtime-gate.json")
        if (
            not g["passed"]
            or g["audit_sha256"] != fingerprint_file(root / "preflight-audit.json").sha256
            or g["freeze_sha256"] != fingerprint_file(root / "frozen.json").sha256
        ):
            raise ValueError("Matching passed audited preflight gate required")
    (root / phase).mkdir()
    for name in ("graphs", "density"):
        (root / phase / name).mkdir()
    binding = read_json(root / "binding.json")
    lookup = {e["key"]: e for e in binding["observations"]}
    start = time.monotonic()
    graph_results, density_results = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for fn, jobs, result_list in (
            (run_graph, graph_jobs(binding, phase), graph_results),
            (run_density, density_jobs(binding, phase), density_results),
        ):
            args = [(str(root), phase, j, lookup[j["key"]], config) for j in jobs]
            for i, result in enumerate(pool.map(fn, args)):
                result_list.append(result)
                if i % 100 == 0 or i == len(args) - 1:
                    print(
                        json.dumps(
                            {
                                "phase": phase,
                                "kind": "graphs" if fn == run_graph else "density",
                                "completed": i + 1,
                                "total": len(args),
                            }
                        ),
                        flush=True,
                    )
                    if (
                        retained_bytes(root) > config.artifact_cap_bytes
                        or shutil.disk_usage(root).free < config.min_free_bytes
                    ):
                        raise ValueError("Approved storage/free-space bound exceeded")
                    if time.monotonic() - start > config.max_projected_seconds:
                        raise ValueError("Approved execution-time bound exceeded")
    evaluations = 2 * len(graph_results) + len(density_results)
    if evaluations != frozen["counts"][phase]:
        raise ValueError("Frozen evaluation accounting differs")
    inventory = {
        str(p.relative_to(root / phase)): fingerprint_file(p).sha256
        for p in sorted((root / phase).rglob("*"))
        if p.is_file()
    }
    write_json(inventory, root / f"{phase}-inventory.json")
    result = {
        "phase": phase,
        "workers": workers,
        "evaluations": evaluations,
        "graphs": graph_results,
        "density": density_results,
        "seconds": time.monotonic() - start,
    }
    write_json(result, root / f"{phase}-execution.json")
    return {
        "evaluations": evaluations,
        "graphs": len(graph_results),
        "density_fits": len(density_results),
        "seconds": result["seconds"],
    }


def gate(root, staging_bytes):
    frozen, config = check_frozen(root)
    if type(staging_bytes) is not int or staging_bytes < 0:
        raise ValueError("Measured nonnegative staging bytes required")
    audit = read_json(root / "preflight-audit.json")
    execution = read_json(root / "preflight-execution.json")
    binding = read_json(root / "binding.json")
    worst = {}
    missing = []
    for n in binding["sizes"]:
        for target in (2, 4):
            times = [
                p["seconds"]
                for g in execution["graphs"]
                if g["n"] == n
                for p in g["partitions"]
                if p["target_k"] == target
                and p["status"] == "available"
                and p.get("method") == "spectral"
            ]
            if not times:
                missing.append([n, target])
            worst[n, target] = max(times, default=0)
    total = 0
    lookup = {e["key"]: e for e in binding["observations"]}
    for job in graph_jobs(binding, "main"):
        n = lookup[job["key"]]["n"]
        base = max(
            g["seconds"] - sum(p.get("seconds", 0) for p in g["partitions"])
            for g in execution["graphs"]
            if g["n"] == n
        )
        total += max(0, base) + sum(worst[n, k] for k in (2, 4))
    for job in density_jobs(binding, "main"):
        n = lookup[job["key"]]["n"]
        total += max(d["seconds"] for d in execution["density"] if d["n"] == n)
    ratio = frozen["counts"]["main"] / frozen["counts"]["preflight"]
    # Conservative single-worker projection includes the independent audit.
    seconds = 1.25 * (total + audit["seconds"] * ratio)
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
        and seconds < config.max_projected_seconds
        and projected <= config.artifact_cap_bytes
        and free >= config.min_free_bytes,
        "projected_main_seconds_conservative": seconds,
        "projected_bytes": projected,
        "staging_bytes": staging_bytes,
        "staging_copies_counted": 2,
        "free_bytes": free,
        "missing_solver_timing_strata": missing,
        "audit_sha256": fingerprint_file(root / "preflight-audit.json").sha256,
        "freeze_sha256": fingerprint_file(root / "frozen.json").sha256,
    }
    write_json(result, root / "storage-runtime-gate.json")
    return result
