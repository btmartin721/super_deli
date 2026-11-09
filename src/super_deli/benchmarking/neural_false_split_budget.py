"""Measured law/draw-count resource projection; no parallel-speed credit."""

import shutil
from pathlib import Path

from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_false_split_state import StudyLock, atomic_json
from super_deli.benchmarking.neural_false_split_study import all_jobs, check_frozen, stop
from super_deli.benchmarking.neural_study import retained_bytes
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json


def stratum(task):
    return f"{task['case']}:N{task['n']}:B{task['n_simulations']}"


def project(root, frozen, config, execution, audit, *, staging_bytes):
    require(type(staging_bytes) is int and staging_bytes >= 0, "Measured staging bytes")
    jobs = {j["task"]["key"]: j for j in all_jobs(root, config, "preflight")}
    per_key = {}
    for row in audit["records"]:
        per_key[row["key"]] = per_key.get(row["key"], 0) + row.get("seconds", 0)
    maxima = {}
    for row in execution["records"]:
        task = jobs[row["key"]]["task"]
        name = stratum(task)
        cell = maxima.setdefault(name, dict(seconds=0, audit_seconds=0, bytes=0))
        cell["seconds"] = max(cell["seconds"], row["seconds"])
        cell["audit_seconds"] = max(cell["audit_seconds"], per_key.get(row["key"], 0))
        cell["bytes"] = max(cell["bytes"], retained_bytes(root / "preflight" / row["key"]))
    main = all_jobs(root, config, "main")
    missing = sorted({stratum(j["task"]) for j in main} - maxima.keys())
    extra = max(0, audit["seconds"] - sum(per_key.values()))
    projected_seconds = sum(
        maxima.get(stratum(j["task"]), {}).get("seconds", 0)
        + maxima.get(stratum(j["task"]), {}).get("audit_seconds", 0)
        for j in main
    ) + extra * len(main) / len(jobs)
    projected_bytes = sum(maxima.get(stratum(j["task"]), {}).get("bytes", 0) for j in main)
    seconds = 1.25 * projected_seconds
    size = retained_bytes(root) + 2 * staging_bytes + int(1.25 * projected_bytes) + 128 * 1024**2
    free = shutil.disk_usage(root).free
    representative = (
        execution["completed"]
        and execution["unknown_counter_units"] == 0
        and execution["attempted"] == frozen["counts"]["preflight"]["workflows"]
        and execution["neural_fits"] == frozen["counts"]["preflight"]["neural_fits"]
        and audit["verified_optimizer_updates"] == frozen["counts"]["preflight"]["verified_updates"]
    )
    return dict(
        passed=not missing
        and representative
        and seconds < config.max_projected_seconds
        and size <= config.artifact_cap_bytes
        and free >= config.min_free_bytes,
        representative_preflight=representative,
        missing_strata=missing,
        measured_maxima=maxima,
        projected_main_audit_seconds=seconds,
        projected_bytes=size,
        staging_bytes=staging_bytes,
        staging_copies_counted=2,
        free_bytes=free,
        accounting="Full law/N/B strata; serial sums plus audit; 25% allowance and 128MiB reserve",
        freeze_sha256=fingerprint_file(root / "frozen.json").sha256,
        execution_sha256=fingerprint_file(root / "preflight-execution.json").sha256,
        audit_sha256=fingerprint_file(root / "preflight-audit.json").sha256,
    )


def gate(root, source, *, staging_bytes):
    from super_deli.benchmarking.view_weight_archive import check_binding

    root, source = Path(root), Path(source)
    frozen, config = check_frozen(root)
    with StudyLock(root):
        try:
            require(not (root / "STOP.json").exists(), "Retained guard requires amendment")
            check_binding(source, read_json(root / "replay-binding.json"))
            e, a = (
                read_json(root / p) for p in ("preflight-execution.json", "preflight-audit.json")
            )
            require(
                a["passed"]
                and a["execution_sha256"]
                == fingerprint_file(root / "preflight-execution.json").sha256,
                "Matching independently audited preflight",
            )
            value = project(root, frozen, config, e, a, staging_bytes=staging_bytes)
            atomic_json(root / "storage-runtime-gate.json", value)
            if not value["passed"]:
                stop(root, {"reason": "Measured preflight resource gate", **value})
            return value
        except Exception as exc:
            stop(root, exc)
            raise
