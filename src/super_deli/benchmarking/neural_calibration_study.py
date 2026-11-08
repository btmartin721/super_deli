"""Shared immutable lifecycle and per-law full-settings storage/runtime gate."""

import json
import shutil
from zipfile import ZipFile

from super_deli.benchmarking import neural_lifecycle as engine
from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_study import retained_bytes
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def project(root, frozen, config, audit, execution, design, staging_bytes):
    from super_deli.benchmarking.neural_calibration_config import tasks

    grid = {t["key"]: t for t in tasks(config, "preflight")}
    times = {}
    sizes = {}
    audits = {}
    point_times = {"neural": 0.0, "pca4": 0.0}
    for row in execution["records"]:
        case = grid[row["key"]]["case"]
        times[case] = max(times.get(case, 0), row["seconds"])
        sizes[case] = max(sizes.get(case, 0), retained_bytes(root / "preflight" / row["key"]))
        for arm in point_times:
            path = root / "preflight" / row["key"] / arm / "calibration/pipeline_artifacts.zip"
            with ZipFile(path) as archive:
                for name in archive.namelist():
                    if name.endswith("/run/summary.json"):
                        summary = json.loads(archive.read(name))
                        point_times[arm] = max(point_times[arm], summary["point_pipeline_seconds"])
    for row in audit["records"]:
        case = grid[row["key"]]["case"]
        audits[case] = audits.get(case, 0) + row["seconds"]
    main = tasks(config, "main")
    missing = sorted({t["case"] for t in main} - times.keys())
    extra = max(audit["seconds"] - sum(r["seconds"] for r in audit["records"]), 0)
    repeat = len(main) / len(grid)
    replay = len(read_json(root / "replay-binding.json")["replays"])
    # Actual full point-pipeline maxima bound replays; whole per-dataset audit
    # maxima add conservative replay/source-check overhead without parallel credit.
    serial = sum(times.get(t["case"], 0) + audits.get(t["case"], 0) for t in main) + extra * repeat
    serial += replay * (sum(point_times.values()) + max(audits.values()) + extra)
    b = sum(sizes.get(t["case"], 0) for t in main) + replay * max(sizes.values())
    seconds = 1.25 * serial
    size = int(retained_bytes(root) + 2 * staging_bytes + 1.25 * b + 128 * 1024**2)
    free = shutil.disk_usage(root).free
    complete = (
        execution["attempted"] == frozen["counts"]["preflight"]["workflows"]
        and execution["neural_fits"] == frozen["counts"]["preflight"]["neural_fits"]
    )
    result = dict(
        passed=not missing
        and complete
        and seconds < config.max_projected_seconds
        and size <= config.artifact_cap_bytes
        and free >= config.min_free_bytes,
        projected_main_audit_seconds=seconds,
        projected_bytes=size,
        staging_bytes=staging_bytes,
        staging_copies_counted=2,
        free_bytes=free,
        missing_strata=missing,
        representative_preflight=complete,
        replay_point_pipeline_maxima=point_times,
        accounting=(
            "Per-law full-B=99 execution/audit and full point-pipeline replay maxima; "
            "whole-dataset audit overhead per replay; serial sum, 25% allowance, "
            "128MiB reserve; no parallel speed credit"
        ),
        freeze_sha256=fingerprint_file(root / "frozen.json").sha256,
        audit_sha256=fingerprint_file(root / "preflight-audit.json").sha256,
        execution_sha256=fingerprint_file(root / "preflight-execution.json").sha256,
    )
    write_json(result, root / "storage-runtime-gate.json")
    if not result["passed"]:
        write_json(dict(reason="Calibration resource gate", **result), root / "STOP.json")
    return result


def design():
    from super_deli.benchmarking.neural_calibration_audit import audit_replay
    from super_deli.benchmarking.neural_calibration_config import (
        ARMS,
        CASES,
        ROOTS,
        NeuralCalibrationStudy,
        counts,
        model_for,
        tasks,
    )
    from super_deli.models.separate_neural import SEPARATE_TRAINING

    return dict(
        config_type=NeuralCalibrationStudy,
        arms=ARMS,
        native_arms=("neural",),
        counts=counts,
        model_for=model_for,
        tasks=tasks,
        audit_replay=audit_replay,
        runner="super_deli.benchmarking.neural_calibration_run",
        auditor="super_deli.benchmarking.neural_calibration_audit",
        serial_runtime_guard=True,
        resource_gate=project,
        training=SEPARATE_TRAINING,
        settings=dict(
            cases=CASES,
            rng_roots=ROOTS,
            n=64,
            neural_seed=0,
            mixture_seed=71323,
            raw_features=[4, 8],
            n_simulations=99,
            primary_rates=16,
            alpha=0.05,
            missingness="error",
            family="tied",
            statistic="Maximum eligible BIC improvement over K=1",
            raw_null="Joint raw full Gaussian mean/covariance ML, before any learning",
            stability="disabled",
            interpretation="Initial N=64 pilot; ten independent datasets per law",
        ),
    )


def freeze(root, config):
    return engine.freeze(root, config, design())


def check_frozen(root):
    return engine.check_frozen(root, design())


def run_phase(root, source, phase, *, workers=1):
    return engine.run_phase(root, source, phase, workers=workers, design=design())


def audit_phase(root, source, phase):
    return engine.audit_phase(root, source, phase, design())


def gate(root, *, staging_bytes):
    require(type(staging_bytes) is int and staging_bytes >= 0, "Measured staging required")
    return engine.gate(root, staging_bytes=staging_bytes, design=design())
