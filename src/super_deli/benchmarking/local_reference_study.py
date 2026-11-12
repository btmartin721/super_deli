"""The unchanged sealed lifecycle with class-local archival jobs and measured gates."""

from pathlib import Path

from super_deli.benchmarking.local_reference_config import (
    LocalReferenceStudy,
    counts,
    settings,
    tasks,
)
from super_deli.benchmarking.local_reference_source import check_source
from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_false_split_state import atomic_json
from super_deli.benchmarking.neural_false_split_study import resource_guard, stop
from super_deli.benchmarking.neural_study import retained_bytes
from super_deli.benchmarking.residual_config import model_for
from super_deli.benchmarking.residual_study import declaration as old_declaration
from super_deli.benchmarking.sealed_calibration_study import SealedCalibrationStudy
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json
from super_deli.models.residual_config import TRAINING


def declaration(root, config):
    result = old_declaration(root, config)
    result["seed_registry_sha256"] = fingerprint_file(root / "seed-registry.json").sha256
    return result


def jobs(config, phase, binding):
    result = [dict(kind="fresh", task=t) for t in tasks(config, phase)]
    for entry in binding["observations"]:
        t = entry["task"]
        selected = (
            (phase == "preflight")
            if config.fixture
            else ((t["replicate"] == 0) == (phase == "preflight"))
        )
        if selected:
            result.append(
                dict(kind="archive", entry=entry, task={**t, "key": "archive_" + t["key"]})
            )
    if phase == "main":
        result.extend(
            dict(kind="replay", entry=e, task=dict(key=e["key"], case="legacy", n=e["n"]))
            for e in binding["replays"]
        )
    return result


class LocalReferenceEngine(SealedCalibrationStudy):
    def binding(self):
        value = read_json(self.root / "replay-binding.json")
        check_source(self.source, value)
        return value


def engine(root, source):
    from super_deli.benchmarking.local_reference_audit import audit_unit
    from super_deli.benchmarking.local_reference_run import run_unit

    design = dict(
        config_type=LocalReferenceStudy,
        arms=("residual",),
        model_for=model_for,
        declaration_base=declaration,
        counts=counts,
        training=TRAINING,
        settings=settings(),
    )
    return LocalReferenceEngine(
        root, source, design, jobs=jobs, runner=run_unit, auditor=audit_unit
    )


def seed_registry(config):
    from super_deli.benchmarking.abstention_config import cohort_names
    from super_deli.benchmarking.abstention_data import parameters

    records = []
    paths = []
    for phase in ("preflight", "main"):
        for task in tasks(config, phase):
            for cohort in cohort_names(task):
                p = parameters(task, cohort)
                seeds = {k: v for k, v in p.items() if k.endswith("_seed")}
                records.append(
                    dict(phase=phase, key=task["key"], cohort=cohort, n=p["n"], seeds=seeds)
                )
                paths.extend(tuple(v) for v in seeds.values())
    require(
        len(paths) == len(set(paths)) and all(p[:2] == (20261112, 740) for p in paths),
        "New disjoint immutable seed domains",
    )
    return dict(
        passed=True,
        materialized_before_generation=True,
        cohorts=len(records),
        seed_paths=len(paths),
        records=records,
    )


def gate(root, source, *, staging_bytes):
    import shutil

    root = Path(root)
    study = engine(root, source)
    _, config = study.check()
    study.binding()
    require(type(staging_bytes) is int and staging_bytes >= 0, "Measured staging evidence")
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
    lookup = {j["task"]["key"]: j for j in study.jobs(config, "preflight")}
    for row in execution["records"]:
        job = lookup[row["key"]]
        t = job["task"]
        measured[(job["kind"], t["case"], t["n"])] = dict(
            seconds=row["seconds"] + times[row["key"]],
            bytes=retained_bytes(root / "preflight" / t["key"]),
        )
    timing = resource_guard(root, config, "preflight")
    overhead = max(0, timing["controller_seconds"] - execution["serial_job_seconds"]) + max(
        0, timing["audit_seconds"] - sum(times.values())
    )
    main = study.jobs(config, "main")
    estimates = []
    for job in main:
        t = job["task"]
        estimates.append(
            max(measured.values(), key=lambda v: v["seconds"])
            if job["kind"] == "replay"
            else measured[(job["kind"], t["case"], t["n"])]
        )
    # Replays get the largest complete fresh/archive unit for BOTH time and space.
    sec = sum(
        v["seconds"] for j, v in zip(main, estimates, strict=True) if j["kind"] != "replay"
    ) + sum(j["kind"] == "replay" for j in main) * max(v["seconds"] for v in measured.values())
    space = sum(
        v["bytes"] for j, v in zip(main, estimates, strict=True) if j["kind"] != "replay"
    ) + sum(j["kind"] == "replay" for j in main) * max(v["bytes"] for v in measured.values())
    seconds = 1.25 * (sec + overhead * len(main) / len(lookup))
    projected = retained_bytes(root) + 2 * staging_bytes + 128 * 1024**2 + int(1.25 * space)
    expected = counts(config, "preflight")
    representative = (
        execution["completed"]
        and execution["attempted"] == expected["workflows"]
        and execution["neural_fits"] == expected["neural_fits"]
        and execution["unknown_counter_units"] == execution["failed_points"] == 0
        and audit["verified_optimizer_updates"] == expected["verified_updates"]
        and sum(r["policy_evaluations"] for r in audit["records"]) == expected["policy_evaluations"]
        and sum(r["oracle_evaluations"] for r in audit["records"]) == expected["oracle_evaluations"]
    )
    free = shutil.disk_usage(root).free
    value = dict(
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
        measured_strata=[dict(kind=k[0], case=k[1], n=k[2], **v) for k, v in measured.items()],
        freeze_sha256=fingerprint_file(root / "frozen.json").sha256,
        audit_sha256=fingerprint_file(root / "preflight-audit.json").sha256,
        execution_sha256=fingerprint_file(root / "preflight-execution.json").sha256,
    )
    atomic_json(root / "storage-runtime-gate.json", value)
    if not value["passed"]:
        stop(root, "Measured class-local storage/runtime gate")
    return value
