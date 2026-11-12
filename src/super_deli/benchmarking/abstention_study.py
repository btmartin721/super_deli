"""Approved source-frozen abstention design on the unchanged sealed-unit lifecycle."""

from pathlib import Path

from super_deli.benchmarking.abstention_config import AbstentionStudy, counts, settings, tasks
from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_false_split_state import atomic_json
from super_deli.benchmarking.neural_false_split_study import resource_guard, stop
from super_deli.benchmarking.neural_study import retained_bytes
from super_deli.benchmarking.residual_config import model_for
from super_deli.benchmarking.residual_study import declaration as original_declaration
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json
from super_deli.models.residual_config import TRAINING


def declaration(root, config):
    value = original_declaration(root, config)
    value["seed_registry_sha256"] = fingerprint_file(root / "seed-registry.json").sha256
    return value


def jobs(config, phase, binding):
    result = [dict(kind="fresh", task=t) for t in tasks(config, phase)]
    if phase == "main":
        result += [
            dict(kind="replay", task=dict(key=e["key"], case="legacy", n=e["n"]), entry=e)
            for e in binding["replays"]
        ]
    return result


def engine(root, source):
    from super_deli.benchmarking.abstention_audit import audit_unit
    from super_deli.benchmarking.abstention_run import run_unit
    from super_deli.benchmarking.sealed_calibration_study import SealedCalibrationStudy

    design = dict(
        config_type=AbstentionStudy,
        arms=("residual",),
        model_for=model_for,
        declaration_base=declaration,
        counts=counts,
        training=TRAINING,
        settings=settings(),
    )
    return SealedCalibrationStudy(
        root, source, design, jobs=jobs, runner=run_unit, auditor=audit_unit
    )


def gate(root, source, *, staging_bytes):
    import shutil

    require(type(staging_bytes) is int and staging_bytes >= 0, "Measured staging evidence")
    root = Path(root)
    study = engine(root, source)
    frozen, config = study.check()
    study.binding()
    execution, audit = (
        read_json(root / "preflight-execution.json"),
        read_json(root / "preflight-audit.json"),
    )
    require(
        audit["passed"]
        and audit["execution_sha256"] == fingerprint_file(root / "preflight-execution.json").sha256,
        "Complete matching preflight audit",
    )
    times = {r["key"]: r["seconds"] for r in audit["records"]}
    measured = {}
    for job in study.jobs(config, "preflight"):
        t = job["task"]
        row = next(r for r in execution["records"] if r["key"] == t["key"])
        measured[(t["case"], t["n"])] = dict(
            seconds=row["seconds"] + times[t["key"]],
            bytes=retained_bytes(root / "preflight" / t["key"]),
        )
    timing = resource_guard(root, config, "preflight")
    overhead = max(0, timing["controller_seconds"] - execution["serial_job_seconds"]) + max(
        0, timing["audit_seconds"] - sum(times.values())
    )
    main = study.jobs(config, "main")
    fresh = [j["task"] for j in main if j["kind"] == "fresh"]
    replay = len(main) - len(fresh)
    # Charge each read-only replay as an entire maximum-cost fresh training unit.
    seconds = 1.25 * (
        sum(measured[(t["case"], t["n"])]["seconds"] for t in fresh)
        + replay * max(v["seconds"] for v in measured.values())
        + overhead * len(main) / len(measured)
    )
    projected = (
        retained_bytes(root)
        + 2 * staging_bytes
        + 128 * 1024**2
        + int(
            1.25
            * (
                sum(measured[(t["case"], t["n"])]["bytes"] for t in fresh)
                + replay * max(v["bytes"] for v in measured.values())
            )
        )
    )
    expected = counts(config, "preflight")
    representative = (
        execution["completed"]
        and execution["attempted"] == expected["workflows"]
        and execution["neural_fits"] == expected["neural_fits"]
        and execution["failed_points"] == execution["unknown_counter_units"] == 0
        and audit["verified_optimizer_updates"] == expected["verified_updates"]
        and sum(r["policy_evaluations"] for r in audit["records"]) == expected["policy_evaluations"]
    )
    free = shutil.disk_usage(root).free
    result = dict(
        passed=representative
        and seconds < config.max_projected_seconds
        and projected <= config.artifact_cap_bytes
        and free >= config.min_free_bytes,
        representative=representative,
        projected_main_audit_seconds=seconds,
        projected_bytes=projected,
        staging_bytes=staging_bytes,
        staging_copies_counted=2,
        free_bytes=free,
        freeze_sha256=fingerprint_file(root / "frozen.json").sha256,
        audit_sha256=fingerprint_file(root / "preflight-audit.json").sha256,
        execution_sha256=fingerprint_file(root / "preflight-execution.json").sha256,
        measured_strata=[dict(case=k[0], n=k[1], **v) for k, v in measured.items()],
    )
    atomic_json(root / "storage-runtime-gate.json", result)
    if not result["passed"]:
        stop(root, "Measured abstention storage/runtime gate")
    return result
