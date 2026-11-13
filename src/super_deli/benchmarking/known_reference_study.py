"""Sealed known-only parent/fold study with separately measured full-settings gates."""

from pathlib import Path

from super_deli.benchmarking.known_reference_config import (
    KnownReferenceStudy,
    counts,
    model_for,
    replays,
    settings,
    tasks,
)
from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_false_split_state import atomic_json
from super_deli.benchmarking.neural_false_split_study import resource_guard, stop
from super_deli.benchmarking.neural_study import retained_bytes
from super_deli.benchmarking.residual_study import declaration as original_declaration
from super_deli.benchmarking.sealed_calibration_study import SealedCalibrationStudy
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json
from super_deli.models.residual_config import TRAINING


def declaration(root, config):
    result = original_declaration(root, config)
    result["seed_registry_sha256"] = fingerprint_file(root / "seed-registry.json").sha256
    return result


def jobs(config, phase, binding):
    require(binding["fixture"] == config.fixture, "Source/scientific fixture agreement")
    result = [dict(kind="fresh", task=t) for t in tasks(config, phase)]
    for item in replays(config, phase):
        parent, h = item["parent"], item["heldout"]
        result.append(
            dict(
                kind="fold_replay",
                task={**parent, "key": f"replay_{parent['key']}_held{h}"},
                parent_task=parent,
                heldout=h,
            )
        )
    require(len(result) == counts(config, phase)["work_units"], "Exact parent/replay schedule")
    return result


def engine(root, source):
    from super_deli.benchmarking.known_reference_audit import audit_unit
    from super_deli.benchmarking.known_reference_run import run_unit

    design = dict(
        config_type=KnownReferenceStudy,
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

    root = Path(root)
    study = engine(root, source)
    _, config = study.check()
    study.binding()
    require(type(staging_bytes) is int and staging_bytes >= 0, "Measured staging bytes")
    execution, audit = (
        read_json(root / "preflight-execution.json"),
        read_json(root / "preflight-audit.json"),
    )
    expected = counts(config, "preflight")
    require(
        audit["passed"]
        and audit["execution_sha256"] == fingerprint_file(root / "preflight-execution.json").sha256,
        "Matching complete independent preflight",
    )
    representative = (
        execution["completed"]
        and execution["attempted"] == expected["workflows"]
        and execution["neural_fits"] == expected["neural_fits"]
        and audit["verified_optimizer_updates"] == expected["verified_updates"]
        and execution["unknown_counter_units"] == execution["failed_points"] == 0
    )
    require(representative, "Exact representative preflight fit/update budgets")
    lookup = {j["task"]["key"]: j for j in study.jobs(config, "preflight")}
    times = {r["key"]: r["seconds"] for r in audit["records"]}
    measured = {}
    for row in execution["records"]:
        j = lookup[row["key"]]
        t = j["task"]
        measured[j["kind"], t["case"], t["n"]] = dict(
            seconds=row["seconds"] + times[t["key"]],
            bytes=retained_bytes(root / "preflight" / t["key"]),
        )
    timing = resource_guard(root, config, "preflight")
    overhead = max(0, timing["controller_seconds"] - execution["serial_job_seconds"]) + max(
        0, timing["audit_seconds"] - sum(times.values())
    )
    estimates = []
    for j in study.jobs(config, "main"):
        t = j["task"]
        key = j["kind"], t["case"], t["n"]
        if j["kind"] == "fresh":
            require(key in measured, "Every law/N independently measured")
            estimates.append(measured[key])
        else:
            peers = [v for k, v in measured.items() if k[0] == j["kind"]]
            require(peers, "Every replay type independently measured")
            estimates.append(
                dict(seconds=max(v["seconds"] for v in peers), bytes=max(v["bytes"] for v in peers))
            )
    seconds = 1.25 * (
        sum(v["seconds"] for v in estimates) + overhead * len(estimates) / len(lookup)
    )
    projected = (
        retained_bytes(root)
        + 2 * staging_bytes
        + 128 * 1024**2
        + int(1.25 * sum(v["bytes"] for v in estimates))
    )
    free = shutil.disk_usage(root).free
    result = dict(
        passed=seconds < config.max_projected_seconds
        and projected <= config.artifact_cap_bytes
        and free >= config.min_free_bytes,
        representative=representative,
        projected_main_audit_seconds=seconds,
        projected_bytes=projected,
        staging_bytes=staging_bytes,
        staging_copies_counted=2,
        free_bytes=free,
        measured_strata=[dict(kind=k[0], case=k[1], n=k[2], **v) for k, v in measured.items()],
        freeze_sha256=fingerprint_file(root / "frozen.json").sha256,
        audit_sha256=fingerprint_file(root / "preflight-audit.json").sha256,
        execution_sha256=fingerprint_file(root / "preflight-execution.json").sha256,
    )
    atomic_json(root / "storage-runtime-gate.json", result)
    if not result["passed"]:
        stop(root, "Measured known-reference runtime/storage gate")
    return result
