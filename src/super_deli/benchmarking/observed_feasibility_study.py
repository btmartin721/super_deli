"""Sealed observed-feature jobs with fixed source heads and measured resource gates."""

from pathlib import Path

from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_false_split_state import atomic_json
from super_deli.benchmarking.neural_false_split_study import resource_guard, stop
from super_deli.benchmarking.neural_study import retained_bytes
from super_deli.benchmarking.observed_feasibility_config import (
    ObservedFeasibilityStudy,
    counts,
    detector_replay_tasks,
    model_for,
    settings,
    tasks,
)
from super_deli.benchmarking.observed_feasibility_source import check_source
from super_deli.benchmarking.residual_study import declaration as original_declaration
from super_deli.benchmarking.sealed_calibration_study import SealedCalibrationStudy
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json


def declaration(root, config):
    value = original_declaration(root, config)
    value["seed_registry_sha256"] = fingerprint_file(root / "seed-registry.json").sha256
    return value


def jobs(config, phase, binding):
    lookup = {
        (e["task"]["case"], e["task"]["n"], e["task"]["replicate"]): e
        for e in binding["observations"]
    }
    require(
        len(lookup) == len(binding["observations"]), "One source observation per case/N/replicate"
    )
    result = []
    for task in tasks(config, phase):
        key = (task["case"], task["n"], task["replicate"])
        require(key in lookup, "Required archived reference heads")
        result.append(dict(kind="fresh", task=task, entry=lookup[key]))
    for task in detector_replay_tasks(config, phase):
        result.append(
            dict(
                kind="detector_replay",
                task={**task, "key": "detector_replay_" + task["key"]},
                original_key=task["key"],
            )
        )
    selected = binding["replays"]
    if phase == "preflight":
        selected = (
            selected[:1]
            if config.fixture
            else [
                e
                for e in selected
                if (e["case"], e["n"], e["arm"])
                in (("linear_weak", 64, "logistic"), ("combined", 256, "residual_0"))
            ]
        )
    for entry in selected:
        result.append(
            dict(
                kind="frozen_replay",
                task=dict(
                    key=entry["key"], case=entry["case"], n=entry["n"], fixture=config.fixture
                ),
                entry=entry,
            )
        )
    require(len(result) == counts(config, phase)["workflows"], "Exact frozen work-unit schedule")
    return result


class ObservedFeasibilityEngine(SealedCalibrationStudy):
    def binding(self):
        value = read_json(self.root / "replay-binding.json")
        check_source(self.source, value)
        return value

    def execution(self, config, phase):
        result = super().execution(config, phase)
        result.update(
            detector_fits=sum(r.get("detector_fits", 0) for r in result["records"]),
            trees=sum(r.get("trees", 0) for r in result["records"]),
        )
        return result


def engine(root, source):
    from super_deli.benchmarking.observed_feasibility_audit import audit_unit
    from super_deli.benchmarking.observed_feasibility_run import run_unit

    design = dict(
        config_type=ObservedFeasibilityStudy,
        arms=("diagnostic",),
        model_for=model_for,
        declaration_base=declaration,
        counts=counts,
        settings=settings(),
        training="One hundred frozen boosted trees; zero neural retraining",
    )
    return ObservedFeasibilityEngine(
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
    require(
        audit["passed"]
        and audit["execution_sha256"] == fingerprint_file(root / "preflight-execution.json").sha256,
        "Complete matching preflight audit",
    )
    expected = counts(config, "preflight")
    require(
        execution["completed"]
        and execution["attempted"] == expected["workflows"]
        and execution["detector_fits"] == expected["detector_fits"]
        and execution["trees"] == expected["trees"]
        and execution["neural_fits"]
        == execution["failed_points"]
        == execution["unknown_counter_units"]
        == 0,
        "Exact representative preflight counters",
    )
    require(
        sum(r["detector_fits"] for r in audit["records"]) == expected["detector_fits"]
        and sum(r["trees"] for r in audit["records"]) == expected["trees"],
        "Independently audited detector/tree budget",
    )
    lookup = {j["task"]["key"]: j for j in study.jobs(config, "preflight")}
    times = {r["key"]: r["seconds"] for r in audit["records"]}
    measured = {}
    for row in execution["records"]:
        j = lookup[row["key"]]
        t = j["task"]
        key = (j["kind"], t["case"], t["n"])
        measured[key] = dict(
            seconds=row["seconds"] + times[row["key"]],
            bytes=retained_bytes(root / "preflight" / row["key"]),
        )
    timing = resource_guard(root, config, "preflight")
    overhead = max(0, timing["controller_seconds"] - execution["serial_job_seconds"]) + max(
        0, timing["audit_seconds"] - sum(times.values())
    )
    estimates = []
    for j in study.jobs(config, "main"):
        t = j["task"]
        key = (j["kind"], t["case"], t["n"])
        if j["kind"] == "fresh":
            require(key in measured, "Every law/N has measured preflight")
            estimates.append(measured[key])
        else:
            peers = [v for k, v in measured.items() if k[0] == j["kind"]]
            require(peers, "Every replay type measured")
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
    value = dict(
        passed=seconds < config.max_projected_seconds
        and projected <= config.artifact_cap_bytes
        and free >= config.min_free_bytes,
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
        stop(root, "Measured observed-feature runtime/storage gate")
    return value
