"""Source-frozen, bounded weighting study with audited resource gates."""

import json
import shutil
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from super_deli.benchmarking.matched_study import declaration as legacy_declaration
from super_deli.benchmarking.matched_study import inventory
from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_study import retained_bytes
from super_deli.benchmarking.view_weight_archive import check_binding
from super_deli.benchmarking.view_weight_config import (
    ARMS,
    FRACTIONS,
    ROOTS,
    SEEDS,
    ViewWeightStudy,
    counts,
    model_for,
    tasks,
)
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def declaration(root, config):
    # Reuse only the established runtime/source declaration; overwrite the
    # legacy design, counts and arms with this independently versioned study.
    value = legacy_declaration(root, config)
    value.update(
        settings={
            "fractions": FRACTIONS,
            "neural_seeds": SEEDS,
            "rng_roots": ROOTS,
            "primary_rates": 108,
            "paired_comparisons": 72,
        },
        counts={p: counts(config, p) for p in ("preflight", "main")},
        arms={a: model_for(a, 0).to_dict() for a in ARMS},
        binding_sha256=fingerprint_file(root / "replay-binding.json").sha256,
    )
    return json.loads(json.dumps(value))


def guard(root, config, staging_bytes=0):
    require(
        retained_bytes(root) + staging_bytes <= config.artifact_cap_bytes, "Evidence cap exceeded"
    )
    require(shutil.disk_usage(root).free >= config.min_free_bytes, "Free-space reserve failed")


def freeze(root, config):
    root = Path(root)
    binding = read_json(root / "replay-binding.json")
    require(
        binding["passed"] and binding["fixture"] == config.fixture,
        "Bound source/configuration kind",
    )
    value = declaration(root, config)
    require(value["counts"]["main"]["legacy_replays"] == len(binding["replays"]), "Replay budget")
    guard(root, config)
    with tarfile.open(root / "frozen-sources.tar.gz", "x:gz") as archive:
        package = Path(__file__).parents[1]
        for name in value["source_sha256"]:
            archive.add(package / name, arcname=name)
    write_json(value, root / "frozen.json")
    return {"frozen": True, "counts": value["counts"]}


def check_frozen(root):
    root = Path(root)
    frozen = read_json(root / "frozen.json")
    config = ViewWeightStudy.from_dict(frozen["configuration"])
    require(
        declaration(root, config) == frozen, "Frozen source/configuration/runtime/binding changed"
    )
    return frozen, config


def run_phase(root, source, phase, *, workers=1):
    from super_deli.benchmarking.view_weight_run import run_dataset, run_replay

    root, source = Path(root), Path(source)
    frozen, config = check_frozen(root)
    require(phase in ("preflight", "main"), "Unknown phase")
    require(type(workers) is int and 1 <= workers <= config.max_workers, "Worker limit")
    binding = read_json(root / "replay-binding.json")
    check_binding(source, binding)
    staging = 0
    if phase == "main":
        gate_record = read_json(root / "storage-runtime-gate.json")
        require(
            gate_record["passed"]
            and all(
                gate_record[k] == fingerprint_file(root / p).sha256
                for k, p in (
                    ("freeze_sha256", "frozen.json"),
                    ("audit_sha256", "preflight-audit.json"),
                    ("execution_sha256", "preflight-execution.json"),
                )
            ),
            "Matching audited preflight gate required",
        )
        staging = 2 * gate_record["staging_bytes"]
    require(not (root / "STOP.json").exists(), "Retained guard failure prohibits dependent work")
    guard(root, config, staging)
    out = root / phase
    out.mkdir(exist_ok=False)
    jobs = [
        ("fresh", {"phase_root": str(out), "task": t, "stop_file": str(root / "STOP.json")})
        for t in tasks(config, phase)
    ]
    if phase == "main":
        jobs += [
            (
                "replay",
                {
                    "phase_root": str(out),
                    "entry": e,
                    "archive_root": str(source),
                    "stop_file": str(root / "STOP.json"),
                },
            )
            for e in binding["replays"]
        ]
    start, records, failure = time.monotonic(), [], None
    try:
        # Batches bound in-flight work after a failed guard to at most one peer.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for offset in range(0, len(jobs), workers):
                if (root / "STOP.json").exists():
                    raise ValueError("Numerical guard stopped dependent work")
                guard(root, config, staging)
                batch = jobs[offset : offset + workers]
                pending = [
                    pool.submit(run_dataset if kind == "fresh" else run_replay, job)
                    for kind, job in batch
                ]
                for (kind, _), future in zip(batch, pending, strict=True):
                    records.append({"kind": kind, **future.result()})
                print({"phase": phase, "jobs": len(records), "total": len(jobs)}, flush=True)
                require(
                    time.monotonic() - start < config.max_projected_seconds,
                    "Execution runtime guard",
                )
        require(not (root / "STOP.json").exists(), "Numerical guard stopped dependent work")
        guard(root, config, staging)
    except Exception as exc:
        failure = str(exc)
        if not (root / "STOP.json").exists():
            write_json({"phase": phase, "reason": failure}, root / "STOP.json")
    result = {
        "phase": phase,
        "workers": workers,
        "records": records,
        "seconds": time.monotonic() - start,
        "serial_job_seconds": sum(r["seconds"] for r in records),
        "planned": frozen["counts"][phase]["workflows"],
        "attempted": sum(r["attempted"] for r in records),
        "neural_fits": sum(r["neural_fits"] for r in records),
        "failed_points": sum(r["failed_points"] for r in records),
        "unattempted_jobs": [
            job.get("task", job.get("entry"))["key"] for _, job in jobs[len(records) :]
        ],
        "completed": failure is None,
        "error": failure,
    }
    write_json(inventory(out), root / f"{phase}-inventory.json")
    write_json(result, root / f"{phase}-execution.json")
    if failure:
        raise ValueError(failure)
    check_binding(source, binding)
    return {k: v for k, v in result.items() if k != "records"}


def audit_replay(out, source, entry, cache):
    import numpy as np

    from super_deli.benchmarking.matched_audit import audit_sources
    from super_deli.benchmarking.neural_analysis import arrays
    from super_deli.benchmarking.neural_moons_coordinates import safe_path
    from super_deli.benchmarking.view_weight_audit import audit_arm

    require(read_json(out / "replay.json") == entry, "Planned replay entry")
    original = safe_path(source, entry["path"])
    data = audit_sources(original, entry["task"])
    task = {**entry["task"], "draws": 0}
    rows = read_json(out / "fits.json")
    require(len(rows) == 1, "One replay workflow")
    row = rows[0]
    require(row["exact_numeric_replay"], "Legacy numeric replay passed")
    folder = out / row["arm"]
    result = audit_arm(folder, task, data, row, cache)
    actual, expected = arrays(folder / "point.npz"), arrays(original / entry["arm"] / "point.npz")
    for name, value in expected.items():
        np.testing.assert_array_equal(actual[name], value)
    initial, old = (
        arrays(folder / "optimizer-checks.npz"),
        arrays(original / entry["arm"] / "initial-weights.npz"),
    )
    for name, value in old.items():
        if name.startswith("fit_0_"):
            np.testing.assert_array_equal(
                initial[name.replace("fit_0_", "fit_0_initial_", 1)], value
            )
    return [result]


def audit_phase(root, source, phase):
    from threadpoolctl import threadpool_limits

    from super_deli.benchmarking.view_weight_audit import audit_dataset

    root, source = Path(root), Path(source)
    frozen, config = check_frozen(root)
    require(not (root / "STOP.json").exists(), "Retained guard requires an amendment")
    start = time.monotonic()
    execution = read_json(root / f"{phase}-execution.json")
    require(execution["completed"], "Complete execution required")
    require(execution["planned"] == frozen["counts"][phase]["workflows"], "Frozen workflow budget")
    require(
        execution["attempted"] <= execution["planned"] and not execution["unattempted_jobs"],
        "No replacement workflows or missing jobs",
    )
    require(
        inventory(root / phase) == read_json(root / f"{phase}-inventory.json"),
        "Complete evidence inventory",
    )
    binding = read_json(root / "replay-binding.json")
    check_binding(source, binding)
    cache, records = {}, []
    try:
        with threadpool_limits(limits=1):
            for task in tasks(config, phase):
                folder = root / phase / task["key"]
                require(read_json(folder / "task.json") == task, "Frozen fresh task")
                records.extend(audit_dataset(folder, task, cache=cache))
            if phase == "main":
                for entry in binding["replays"]:
                    records.extend(audit_replay(root / phase / entry["key"], source, entry, cache))
        require(
            sum(r["neural_fits"] for r in records) == execution["neural_fits"],
            "Actual neural denominator",
        )
        guard(root, config)
        require(
            execution["seconds"] + time.monotonic() - start < config.max_projected_seconds,
            "Execution plus independent audit runtime guard",
        )
        result = {
            "passed": True,
            "phase": phase,
            "records": records,
            "neural_fits": execution["neural_fits"],
            "verified_optimizer_updates": sum(r["verified_updates"] for r in records),
            "failed_points": execution["failed_points"],
            "native_draws": sum(r["draws"] for r in records),
            "audit_boundary": (
                "Actual-path NumPy gradient/Adam checks, independent losses, raw laws, "
                "identities, transforms, graph hierarchy, GMM likelihood and denominators. "
                "Failed endpoints retain partial evidence and are not claimed as verified fits."
            ),
            "seconds": time.monotonic() - start,
            "freeze_sha256": fingerprint_file(root / "frozen.json").sha256,
            "execution_sha256": fingerprint_file(root / f"{phase}-execution.json").sha256,
        }
    except Exception as exc:
        result = {
            "passed": False,
            "phase": phase,
            "error": str(exc),
            "seconds": time.monotonic() - start,
        }
        write_json(result, root / f"{phase}-audit.json")
        write_json(result, root / "STOP.json")
        raise
    write_json(result, root / f"{phase}-audit.json")
    return {k: v for k, v in result.items() if k != "records"}


def gate(root, *, staging_bytes):
    root = Path(root)
    frozen, config = check_frozen(root)
    require(type(staging_bytes) is int and staging_bytes >= 0, "Measured staging bytes required")
    audit = read_json(root / "preflight-audit.json")
    execution = read_json(root / "preflight-execution.json")
    require(
        audit["passed"]
        and audit["execution_sha256"] == fingerprint_file(root / "preflight-execution.json").sha256,
        "Audited unchanged preflight required",
    )
    # Five main observations per preflight cell. This deliberately projects
    # all attached preflight refits five times, bounding the main's single set.
    ratio = 1 if config.fixture else 5
    from super_deli.benchmarking.view_weight_config import JOINT

    joint_folders = [
        root / "preflight" / t["key"] / arm for t in tasks(config, "preflight") for arm in JOINT
    ]
    largest_job = max(read_json(p / "execution.json")["seconds"] for p in joint_folders)
    replays = frozen["counts"]["main"]["legacy_replays"]
    serial = max(execution["serial_job_seconds"], execution["seconds"])
    seconds = 1.25 * (ratio * (serial + audit["seconds"]) + replays * largest_job)
    # Storage uses the largest single dataset, not an entire phase, for replays.
    largest_bytes = max(retained_bytes(p) for p in joint_folders)
    projected = (
        retained_bytes(root)
        + 2 * staging_bytes
        + int(1.25 * (ratio * retained_bytes(root / "preflight") + replays * largest_bytes))
        + 64 * 1024**2
    )
    free = shutil.disk_usage(root).free
    result = {
        "passed": seconds < config.max_projected_seconds
        and projected <= config.artifact_cap_bytes
        and free >= config.min_free_bytes,
        "projected_bytes": projected,
        "projected_main_audit_seconds": seconds,
        "staging_bytes": staging_bytes,
        "staging_copies_counted": 2,
        "free_bytes": free,
        "freeze_sha256": fingerprint_file(root / "frozen.json").sha256,
        "audit_sha256": fingerprint_file(root / "preflight-audit.json").sha256,
        "execution_sha256": fingerprint_file(root / "preflight-execution.json").sha256,
    }
    write_json(result, root / "storage-runtime-gate.json")
    if not result["passed"]:
        write_json({"reason": "Measured preflight resource gate", **result}, root / "STOP.json")
    return result
