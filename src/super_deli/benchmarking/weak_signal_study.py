"""Source-frozen archival study on the existing sealed-work-unit lifecycle."""

import platform
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_false_split_state import atomic_json
from super_deli.benchmarking.neural_false_split_study import resource_guard, stop
from super_deli.benchmarking.neural_study import retained_bytes, source_hashes
from super_deli.benchmarking.weak_signal_config import (
    ARMS,
    NUMERICAL,
    WeakSignalStudy,
    counts_from_tasks,
    tasks,
)
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json


def declaration(root, config):
    from super_deli import __version__
    from super_deli.benchmarking.weak_signal_source import check_inputs

    check_inputs(root)
    return dict(
        configuration=config.to_dict(),
        source_sha256=source_hashes(),
        plan_sha256=fingerprint_file(root / "approved-plan.md").sha256,
        python=sys.version,
        platform=platform.platform(),
        package_version=__version__,
        software={name: version(name) for name in NUMERICAL},
    )


def design(root):
    from super_deli.benchmarking.coordinate_config import MixtureSettings

    entries = read_json(root / "replay-binding.json")["entries"]
    return dict(
        config_type=WeakSignalStudy,
        arms=ARMS,
        native_arms=(),
        declaration_base=declaration,
        model_for=lambda arm, seed: MixtureSettings("tied"),
        counts=lambda config, phase: counts_from_tasks(tasks(entries, phase)),
        settings=dict(
            reference_noise_sd=0.15,
            signal_multipliers=[0.5, 1.0, 1.5],
            maximum_k=4,
            recovery="K=4 AND ARI>=0.8",
            truth_starts=1,
            known_prior="Equal marginal; no fixed-capacity assignment",
            neural_training=False,
            new_pvalues=False,
            interpretation="Diagnostic references are not ARI ceilings",
        ),
    )


def jobs(config, phase, binding):
    return [dict(kind="fresh", task=t) for t in tasks(binding["entries"], phase)]


def engine(root, source):
    from super_deli.benchmarking.sealed_calibration_study import SealedCalibrationStudy
    from super_deli.benchmarking.weak_signal_audit import audit_unit
    from super_deli.benchmarking.weak_signal_run import run_unit

    return SealedCalibrationStudy(
        root, source, design(Path(root)), jobs=jobs, runner=run_unit, auditor=audit_unit
    )


def gate(root, source, *, staging_bytes):
    root = Path(root)
    require(type(staging_bytes) is int and staging_bytes >= 0, "Measured staging bytes")
    study = engine(root, source)
    frozen, config = study.check()
    study.binding()
    audit = read_json(root / "preflight-audit.json")
    execution = read_json(root / "preflight-execution.json")
    require(
        audit["passed"]
        and audit["execution_sha256"] == fingerprint_file(root / "preflight-execution.json").sha256,
        "Audited unchanged preflight",
    )

    def key(task):
        return task["case"], task["n"], task["signal_multiplier"]

    measured, audits = {}, {r["key"]: r["seconds"] for r in audit["records"]}
    entries = {j["task"]["key"]: j["task"] for j in study.jobs(config, "preflight")}
    for row in execution["records"]:
        measured[key(entries[row["key"]])] = dict(
            seconds=row["seconds"] + audits[row["key"]],
            bytes=retained_bytes(root / "preflight" / row["key"]),
        )
    main = study.jobs(config, "main")
    require(all(key(j["task"]) in measured for j in main), "Every main stratum measured")
    timing = resource_guard(root, config, "preflight")
    extra = max(0, timing["controller_seconds"] - execution["serial_job_seconds"]) + max(
        0, timing["audit_seconds"] - sum(audits.values())
    )
    seconds = 1.25 * (
        sum(measured[key(j["task"])]["seconds"] for j in main) + extra * len(main) / len(entries)
    )
    size = (
        retained_bytes(root)
        + 2 * staging_bytes
        + 64 * 1024**2
        + int(1.25 * sum(measured[key(j["task"])]["bytes"] for j in main))
    )
    free = shutil.disk_usage(root).free
    passed = (
        execution["completed"]
        and execution["unknown_counter_units"] == 0
        and execution["attempted"] <= frozen["counts"]["preflight"]["workflows"]
        and seconds < config.max_projected_seconds
        and size <= config.artifact_cap_bytes
        and free >= config.min_free_bytes
    )
    result = dict(
        passed=passed,
        projected_main_audit_seconds=seconds,
        projected_bytes=size,
        staging_bytes=staging_bytes,
        free_bytes=free,
        freeze_sha256=fingerprint_file(root / "frozen.json").sha256,
        audit_sha256=fingerprint_file(root / "preflight-audit.json").sha256,
        execution_sha256=fingerprint_file(root / "preflight-execution.json").sha256,
    )
    atomic_json(root / "storage-runtime-gate.json", result)
    if not passed:
        stop(root, "Measured weak-signal resource gate")
    return result
