"""Approved source-frozen residual benchmark on the existing sealed-unit lifecycle."""

import platform
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_false_split_state import atomic_json
from super_deli.benchmarking.neural_false_split_study import resource_guard, stop
from super_deli.benchmarking.neural_study import retained_bytes, source_hashes
from super_deli.benchmarking.residual_config import (
    CASES,
    NAMESPACE,
    SEEDS,
    ResidualStudy,
    counts,
    model_for,
    tasks,
)
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json
from super_deli.models.residual_config import TRAINING


def declaration(root, config):
    from super_deli import __version__

    return dict(
        configuration=config.to_dict(),
        source_sha256=source_hashes(),
        plan_sha256=fingerprint_file(root / "approved-plan.md").sha256,
        python=sys.version,
        platform=platform.platform(),
        package_version=__version__,
        software={
            name: version(name)
            for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl", "torch")
        },
    )


def design():
    return dict(
        config_type=ResidualStudy,
        arms=("residual", "pca4", "neural"),
        model_for=model_for,
        declaration_base=declaration,
        counts=counts,
        training=TRAINING,
        settings=dict(
            cases=CASES,
            rng_namespace=NAMESPACE,
            seeds=SEEDS,
            primary_seed=0,
            primary_rates=40,
            inference="Reference assignment and discovery are separate",
            known_class_evaluation=2048,
            unfamiliar_class_evaluation=512,
            logistic=dict(C=1, solver="lbfgs", max_iter=2000, tol=1e-9),
            stability="disabled",
            significance="disabled",
        ),
    )


def jobs(config, phase, binding):
    result = [dict(kind="fresh", task=t) for t in tasks(config, phase)]
    if phase == "main":
        result += [
            dict(
                kind="replay",
                task=dict(key=e["key"], n=e["original_task"]["n"], case="legacy"),
                entry=e,
            )
            for e in binding["replays"]
        ]
    return result


def engine(root, source):
    from super_deli.benchmarking.residual_audit import audit_unit
    from super_deli.benchmarking.residual_run import run_unit
    from super_deli.benchmarking.sealed_calibration_study import SealedCalibrationStudy

    return SealedCalibrationStudy(
        root, source, design(), jobs=jobs, runner=run_unit, auditor=audit_unit
    )


def gate(root, source, *, staging_bytes):
    root = Path(root)
    require(type(staging_bytes) is int and staging_bytes >= 0, "Measured staging bytes")
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
        "Matching complete preflight audit",
    )
    measured, arm_seconds, arm_bytes = {}, [], []
    audit_times = {r["key"]: r["seconds"] for r in audit["records"]}
    pref = {j["task"]["key"]: j["task"] for j in study.jobs(config, "preflight")}
    for row in execution["records"]:
        t = pref[row["key"]]
        folder = root / "preflight" / t["key"]
        measured[(t["case"], t["n"])] = dict(
            seconds=row["seconds"] + audit_times[t["key"]], bytes=retained_bytes(folder)
        )
        for r in read_json(folder / "fits.json"):
            if r["task"] == "unsupervised_discovery":
                arm_seconds.append(r["seconds"])
                arm_bytes.append(retained_bytes(folder / r["arm"]))
    timing = resource_guard(root, config, "preflight")
    overhead = max(0, timing["controller_seconds"] - execution["serial_job_seconds"]) + max(
        0, timing["audit_seconds"] - sum(audit_times.values())
    )
    main = study.jobs(config, "main")
    fresh = [j["task"] for j in main if j["kind"] == "fresh"]
    replay = len(main) - len(fresh)
    estimate_seconds = sum(measured[(t["case"], t["n"])]["seconds"] for t in fresh) + replay * (
        max(arm_seconds) + max(audit_times.values())
    )
    estimate_bytes = sum(measured[(t["case"], t["n"])]["bytes"] for t in fresh) + replay * (
        max(arm_bytes) + 1024**2
    )
    seconds = 1.25 * (estimate_seconds + overhead * len(main) / len(pref))
    projected = (
        retained_bytes(root) + 2 * staging_bytes + 128 * 1024**2 + int(1.25 * estimate_bytes)
    )
    expected = counts(config, "preflight")
    representative = (
        execution["completed"]
        and execution["attempted"] == expected["workflows"]
        and execution["neural_fits"] == expected["neural_fits"]
        and audit["verified_optimizer_updates"] == expected["verified_updates"]
        and execution["unknown_counter_units"] == 0
    )
    free = shutil.disk_usage(root).free
    result = dict(
        passed=representative
        and seconds < config.max_projected_seconds
        and projected <= config.artifact_cap_bytes
        and free >= config.min_free_bytes,
        projected_main_audit_seconds=seconds,
        projected_bytes=projected,
        staging_bytes=staging_bytes,
        staging_copies_counted=2,
        free_bytes=free,
        representative=representative,
        freeze_sha256=fingerprint_file(root / "frozen.json").sha256,
        audit_sha256=fingerprint_file(root / "preflight-audit.json").sha256,
        execution_sha256=fingerprint_file(root / "preflight-execution.json").sha256,
        measured_strata=[dict(case=k[0], n=k[1], **v) for k, v in measured.items()],
    )
    atomic_json(root / "storage-runtime-gate.json", result)
    if not result["passed"]:
        stop(root, "Measured residual/nonlinear resource gate")
    return result
