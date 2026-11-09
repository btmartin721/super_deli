"""Immutable batched execution with preserved interruptions and cumulative limits."""

import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from super_deli.benchmarking import neural_lifecycle as lifecycle
from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_false_split_config import (
    ARMS,
    CASES,
    NAMESPACE,
    NeuralFalseSplitStudy,
    archived_tasks,
    batch_count,
    batch_tasks,
    counts,
    tasks,
)
from super_deli.benchmarking.neural_false_split_state import (
    StudyLock,
    atomic_json,
    process_alive,
    seal,
    seal_path,
    verify_seal,
)
from super_deli.benchmarking.view_weight_archive import check_binding
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json


def design():
    from super_deli.benchmarking.neural_calibration_config import model_for
    from super_deli.models.separate_neural import SEPARATE_TRAINING

    return dict(
        config_type=NeuralFalseSplitStudy,
        arms=ARMS,
        native_arms=("neural",),
        counts=counts,
        model_for=model_for,
        tasks=tasks,
        training=SEPARATE_TRAINING,
        settings=dict(
            cases=CASES,
            rng_namespace=NAMESPACE,
            n=64,
            neural_seed=0,
            mixture_seed=71323,
            raw_features=[4, 8],
            n_simulations=99,
            diagnostic_draws=999,
            observations_per_law=200,
            batches=8,
            primary_rates=8,
            paired_comparisons=4,
            alpha=0.05,
            family="tied",
            missingness="error",
            stability="disabled",
            statistic="Maximum eligible BIC improvement over K=1",
            raw_null="Joint raw full Gaussian mean/covariance ML, before learning",
            interpretation="N=64 Gaussian false-split validation; no species/selected-K guarantee",
        ),
    )


def freeze(root, config):
    return lifecycle.freeze(Path(root), config, design())


def check_frozen(root):
    return lifecycle.check_frozen(Path(root), design())


def all_jobs(root, config, phase):
    jobs = [dict(kind="fresh", task=t) for t in tasks(config, phase)]
    if phase == "main":
        jobs.extend(archived_tasks(config, read_json(root / "replay-binding.json")))
    return jobs


def group_jobs(root, config, phase, group):
    if phase == "preflight":
        require(group == "preflight", "One fixed preflight group")
        return all_jobs(root, config, phase)
    if group == "extras":
        return archived_tasks(config, read_json(root / "replay-binding.json"))
    require(group.startswith("batch_"), "Fixed main batch or extras required")
    index = int(group.removeprefix("batch_"))
    return [dict(kind="fresh", task=t) for t in batch_tasks(config, index)]


def group_names(config, phase):
    return (
        ["preflight"]
        if phase == "preflight"
        else [*[f"batch_{i:02d}" for i in range(batch_count(config))], "extras"]
    )


def resource_guard(root, config, phase, *, staging_bytes=0):
    lifecycle.guard(root, config, staging_bytes)
    serial = sum(
        read_json(p)["result"]["seconds"] for p in (root / "ledger" / phase).glob("*.json")
    )
    wall = sum(
        r["elapsed"] if r["elapsed"] is not None else max(0, time.monotonic() - r["started"])
        for r in (read_json(p) for p in (root / "state/controllers" / phase).glob("*.json"))
    )
    audit = sum(read_json(p)["seconds"] for p in (root / "audits" / phase).glob("*.json"))
    cap = config.preflight_max_seconds if phase == "preflight" else config.max_projected_seconds
    require(max(serial, wall) + audit < cap, "Cumulative execution plus audit runtime ceiling")
    return dict(serial_job_seconds=serial, controller_seconds=wall, audit_seconds=audit)


def checked_gate(root):
    gate = read_json(root / "storage-runtime-gate.json")
    require(
        gate["passed"]
        and all(
            gate[key] == fingerprint_file(root / path).sha256
            for key, path in (
                ("freeze_sha256", "frozen.json"),
                ("audit_sha256", "preflight-audit.json"),
                ("execution_sha256", "preflight-execution.json"),
            )
        ),
        "Matching independently audited resource gate",
    )
    return 2 * gate["staging_bytes"]


def stop(root, reason):
    if not (root / "STOP.json").exists():
        atomic_json(root / "STOP.json", dict(reason=str(reason)))


def recover_interrupted(root, phase, job):
    key = job["task"]["key"]
    marker = read_json(root / "state/active" / (phase + "_" + key + ".json"))
    require(
        marker["key"] == key and marker["started"] <= time.monotonic(),
        "Recognized interrupted work and monotonic clock",
    )
    require(
        marker["status"] != "running" or not process_alive(marker["pid"]),
        "Never finalize an active orphaned worker",
    )
    elapsed = marker.get("elapsed")
    if elapsed is None:
        elapsed = time.monotonic() - marker["started"]
    result = dict(
        key=key,
        seconds=elapsed,
        attempted=None,
        neural_fits=None,
        failed_points=None,
        reason="Interrupted unit retained without replacement",
    )
    return seal(root, phase, job, result, interrupted=True)


def group_execution(root, config, phase, group):
    records, missing = [], []
    jobs = group_jobs(root, config, phase, group)
    for job in jobs:
        if seal_path(root, phase, job["task"]["key"]).exists():
            value = verify_seal(root, phase, job)
            records.append(
                dict(kind=job["kind"], interrupted=value["interrupted"], **value["result"])
            )
        else:
            missing.append(job["task"]["key"])
    unknown = sum(r["attempted"] is None for r in records)
    return dict(
        phase=phase,
        group=group,
        completed=not missing,
        records=records,
        unattempted_jobs=missing,
        unknown_counter_units=unknown,
        planned=sum(2 * (1 + j["task"]["n_simulations"]) for j in jobs),
        attempted=sum(r["attempted"] or 0 for r in records),
        neural_fits=sum(r["neural_fits"] or 0 for r in records),
        failed_points=sum(r["failed_points"] or 0 for r in records),
        serial_job_seconds=sum(r["seconds"] for r in records),
        freeze_sha256=fingerprint_file(root / "frozen.json").sha256,
    )


def run_group(root, source, phase, group, *, workers=1):
    from super_deli.benchmarking.neural_false_split_run import run_unit

    root, source = Path(root), Path(source)
    _, config = check_frozen(root)
    require(phase in ("preflight", "main"), "Known study phase")
    require(type(workers) is int and 1 <= workers <= config.max_workers, "Worker ceiling")
    jobs = group_jobs(root, config, phase, group)
    names = group_names(config, phase)
    require(group in names, "Prespecified group")
    for previous in names[: names.index(group)]:
        prior = read_json(root / "audits" / phase / (previous + ".json"))
        path = root / "executions" / phase / (previous + ".json")
        require(
            prior["passed"] and prior["execution_sha256"] == fingerprint_file(path).sha256,
            "Earlier fixed group independently audited",
        )
    staging = checked_gate(root) if phase == "main" else 0
    require(not (root / "STOP.json").exists(), "Retained guard prohibits dependent work")
    with StudyLock(root):
        existing = root / "executions" / phase / (group + ".json")
        if existing.exists():
            try:
                check_binding(source, read_json(root / "replay-binding.json"))
                value = group_execution(root, config, phase, group)
                require(read_json(existing) == value, "Verified completed group unchanged")
                return {k: v for k, v in value.items() if k != "records"}
            except Exception as exc:
                stop(root, exc)
                raise
        started = time.monotonic()
        controller = root / "state/controllers" / phase / (uuid.uuid4().hex + ".json")
        atomic_json(controller, dict(started=started, elapsed=None, group=group))
        try:
            check_binding(source, read_json(root / "replay-binding.json"))
            resource_guard(root, config, phase, staging_bytes=staging)
            (root / phase).mkdir(exist_ok=True)
            pending = []
            for job in jobs:
                key = job["task"]["key"]
                if seal_path(root, phase, key).exists():
                    verify_seal(root, phase, job)
                elif (root / phase / key).exists():
                    recover_interrupted(root, phase, job)
                else:
                    pending.append(job)
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for offset in range(0, len(pending), workers):
                    if (root / "PAUSE.json").exists():
                        break
                    require(not (root / "STOP.json").exists(), "Earlier numerical guard")
                    resource_guard(root, config, phase, staging_bytes=staging)
                    chunk = pending[offset : offset + workers]
                    futures = [
                        pool.submit(
                            run_unit,
                            {
                                **job,
                                "phase_root": str(root / phase),
                                "archive_root": str(source),
                                "stop_file": str(root / "STOP.json"),
                            },
                        )
                        for job in chunk
                    ]
                    for job, future in zip(chunk, futures, strict=True):
                        seal(root, phase, job, future.result())
                    print(
                        dict(
                            phase=phase,
                            group=group,
                            units_completed=offset + len(chunk),
                            untouched=len(pending) - offset - len(chunk),
                        ),
                        flush=True,
                    )
                    check_frozen(root)
                    resource_guard(root, config, phase, staging_bytes=staging)
            require(not (root / "STOP.json").exists(), "Retained numerical guard")
            execution = group_execution(root, config, phase, group)
            path = root / "executions" / phase / (group + ".json")
            if execution["completed"]:
                if path.exists():
                    require(read_json(path) == execution, "Verified completed group unchanged")
                else:
                    atomic_json(path, execution)
            check_binding(source, read_json(root / "replay-binding.json"))
            return {k: v for k, v in execution.items() if k != "records"}
        except Exception as exc:
            stop(root, exc)
            raise
        finally:
            atomic_json(
                controller,
                dict(started=started, elapsed=time.monotonic() - started, group=group),
                replace=True,
            )


def audit_group(root, source, phase, group):
    from threadpoolctl import threadpool_limits

    from super_deli.benchmarking.neural_false_split_audit import audit_unit

    root, source = Path(root), Path(source)
    _, config = check_frozen(root)
    path = root / "audits" / phase / (group + ".json")
    with StudyLock(root):
        try:
            require(not (root / "STOP.json").exists(), "Retained guard requires amendment")
            check_binding(source, read_json(root / "replay-binding.json"))
            execution = read_json(root / "executions" / phase / (group + ".json"))
            require(
                execution == group_execution(root, config, phase, group), "Complete immutable group"
            )
            if path.exists():
                old = read_json(path)
                require(
                    old["passed"]
                    and old["execution_sha256"]
                    == fingerprint_file(root / "executions" / phase / (group + ".json")).sha256,
                    "Matching prior audit",
                )
                return old
            started, cache, records = time.monotonic(), {}, []
            with threadpool_limits(limits=1):
                for job in group_jobs(root, config, phase, group):
                    value = verify_seal(root, phase, job)
                    if value["interrupted"]:
                        records.append(
                            dict(
                                key=job["task"]["key"],
                                interrupted=True,
                                neural_fits=0,
                                verified_updates=0,
                                numerical_audit="Unavailable; retained bytes verified only",
                            )
                        )
                    else:
                        rows = audit_unit(
                            root / phase / job["task"]["key"], job, source, cache=cache
                        )
                        records.extend(rows)
            require(
                sum(r["neural_fits"] for r in records) == execution["neural_fits"],
                "Known neural denominator; interrupted prefixes remain unknown",
            )
            result = dict(
                passed=True,
                phase=phase,
                group=group,
                records=records,
                seconds=time.monotonic() - started,
                neural_fits=execution["neural_fits"],
                verified_optimizer_updates=sum(r["verified_updates"] for r in records),
                interrupted_units=execution["unknown_counter_units"],
                freeze_sha256=fingerprint_file(root / "frozen.json").sha256,
                execution_sha256=fingerprint_file(
                    root / "executions" / phase / (group + ".json")
                ).sha256,
            )
            atomic_json(path, result)
            resource_guard(
                root, config, phase, staging_bytes=checked_gate(root) if phase == "main" else 0
            )
            check_binding(source, read_json(root / "replay-binding.json"))
            return {k: v for k, v in result.items() if k != "records"}
        except Exception as exc:
            stop(root, exc)
            raise


def finalize_phase(root, phase):
    root = Path(root)
    with StudyLock(root):
        try:
            return _finalize_phase(root, phase)
        except Exception as exc:
            stop(root, exc)
            raise


def _finalize_phase(root, phase):
    root = Path(root)
    require(phase in ("preflight", "main"), "Known study phase")
    _, config = check_frozen(root)
    require(not (root / "STOP.json").exists(), "Retained guard prohibits finalization")
    executions, audits = [], []
    for group in group_names(config, phase):
        path = root / "executions" / phase / (group + ".json")
        e, a = read_json(path), read_json(root / "audits" / phase / (group + ".json"))
        require(e == group_execution(root, config, phase, group), "All frozen groups complete")
        require(
            a["passed"] and a["execution_sha256"] == fingerprint_file(path).sha256,
            "All fixed groups independently audited",
        )
        executions.append(e)
        audits.append(a)
    timing = resource_guard(
        root, config, phase, staging_bytes=checked_gate(root) if phase == "main" else 0
    )
    e = dict(
        phase=phase,
        completed=True,
        records=[r for e in executions for r in e["records"]],
        planned=counts(config, phase)["workflows"],
        unattempted_jobs=[],
        **timing,
    )
    for name in ("attempted", "neural_fits", "failed_points", "unknown_counter_units"):
        e[name] = sum(x[name] for x in executions)
    path = root / (phase + "-execution.json")
    if path.exists():
        require(read_json(path) == e, "Finalized phase execution unchanged")
    else:
        atomic_json(path, e)
    a = dict(
        passed=True,
        phase=phase,
        records=[r for a in audits for r in a["records"]],
        seconds=sum(x["seconds"] for x in audits),
        neural_fits=e["neural_fits"],
        verified_optimizer_updates=sum(x["verified_optimizer_updates"] for x in audits),
        freeze_sha256=fingerprint_file(root / "frozen.json").sha256,
        execution_sha256=fingerprint_file(root / (phase + "-execution.json")).sha256,
    )
    path = root / (phase + "-audit.json")
    if path.exists():
        require(read_json(path) == a, "Finalized phase audit unchanged")
    else:
        atomic_json(path, a)
    return dict(passed=True, phase=phase, attempted=e["attempted"], **timing)
