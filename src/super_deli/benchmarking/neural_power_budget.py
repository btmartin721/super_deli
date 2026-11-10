"""Measured law/strength/N projection with no parallel-speed credit."""

import shutil
from pathlib import Path

from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_false_split_state import StudyLock, atomic_json
from super_deli.benchmarking.neural_false_split_study import stop
from super_deli.benchmarking.neural_power_study import engine
from super_deli.benchmarking.neural_study import retained_bytes
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json


def stratum(task):
    strength = task.get("signal_multiplier", 1.0)
    return f"{task['case']}:s{strength:g}:N{task['n']}:B{task['n_simulations']}"


def project(root, frozen, config, execution, audit, jobs, *, staging_bytes):
    require(type(staging_bytes) is int and staging_bytes >= 0, "Measured staging required")
    fresh = {j["task"]["key"]: j["task"] for j in jobs["preflight"]}
    per_key, maxima = {}, {}
    for row in audit["records"]:
        per_key[row["key"]] = per_key.get(row["key"], 0) + row.get("seconds", 0)
    for row in execution["records"]:
        key = row["key"]
        name = stratum(fresh[key])
        maxima[name] = dict(
            seconds=row["seconds"],
            audit_seconds=per_key.get(key, 0),
            bytes=retained_bytes(root / "preflight" / key),
        )
    main = jobs["main"]
    missing = sorted({stratum(j["task"]) for j in main} - maxima.keys())
    extra = max(0, audit["seconds"] - sum(per_key.values()))
    seconds = 1.25 * (
        sum(
            maxima.get(stratum(j["task"]), {}).get("seconds", 0)
            + maxima.get(stratum(j["task"]), {}).get("audit_seconds", 0)
            for j in main
        )
        + extra * len(main) / len(fresh)
    )
    size = (
        retained_bytes(root)
        + 2 * staging_bytes
        + 128 * 1024**2
        + int(1.25 * sum(maxima.get(stratum(j["task"]), {}).get("bytes", 0) for j in main))
    )
    representative = (
        execution["completed"]
        and execution["unknown_counter_units"] == 0
        and execution["attempted"] == frozen["counts"]["preflight"]["workflows"]
        and execution["neural_fits"] == frozen["counts"]["preflight"]["neural_fits"]
        and audit["verified_optimizer_updates"] == frozen["counts"]["preflight"]["verified_updates"]
    )
    free = shutil.disk_usage(root).free
    return dict(
        passed=not missing
        and representative
        and seconds < config.max_projected_seconds
        and size <= config.artifact_cap_bytes
        and free >= config.min_free_bytes,
        measured_maxima=maxima,
        missing_strata=missing,
        representative_preflight=representative,
        projected_main_audit_seconds=seconds,
        projected_bytes=size,
        staging_bytes=staging_bytes,
        staging_copies_counted=2,
        free_bytes=free,
        accounting="Full law/strength/N/B strata; serial sums; 25% allowance,128MiB reserve",
        freeze_sha256=fingerprint_file(root / "frozen.json").sha256,
        execution_sha256=fingerprint_file(root / "preflight-execution.json").sha256,
        audit_sha256=fingerprint_file(root / "preflight-audit.json").sha256,
    )


def gate(root, source, *, staging_bytes):
    root = Path(root)
    study = engine(root, source)
    frozen, config = study.check()
    with StudyLock(root):
        try:
            require(not (root / "STOP.json").exists(), "Retained guard requires amendment")
            study.binding()
            e, a = (
                read_json(root / p) for p in ("preflight-execution.json", "preflight-audit.json")
            )
            require(
                a["passed"]
                and a["execution_sha256"]
                == fingerprint_file(root / "preflight-execution.json").sha256,
                "Audited preflight",
            )
            value = project(
                root,
                frozen,
                config,
                e,
                a,
                {p: study.jobs(config, p) for p in ("preflight", "main")},
                staging_bytes=staging_bytes,
            )
            atomic_json(root / "storage-runtime-gate.json", value)
            if not value["passed"]:
                stop(root, {"reason": "Measured resource gate", **value})
            return value
        except Exception as exc:
            stop(root, exc)
            raise
