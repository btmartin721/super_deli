"""Reusable source-frozen paired-calibration execution on sealed whole work units."""

import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from super_deli.benchmarking import neural_lifecycle as lifecycle
from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_false_split_state import (
    StudyLock,
    atomic_json,
    seal,
    seal_path,
    verify_seal,
)
from super_deli.benchmarking.neural_false_split_study import (
    checked_gate,
    recover_interrupted,
    resource_guard,
    stop,
)
from super_deli.benchmarking.view_weight_archive import check_binding
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json


class SealedCalibrationStudy:
    """Inject a frozen design and native worker/auditor without altering old engines."""

    def __init__(self, root, source, design, *, jobs, runner, auditor):
        self.root, self.source = Path(root), Path(source)
        self.design, self.job_factory = design, jobs
        self.runner, self.auditor = runner, auditor

    def freeze(self, config):
        return lifecycle.freeze(self.root, config, self.design)

    def check(self):
        return lifecycle.check_frozen(self.root, self.design)

    def binding(self):
        value = read_json(self.root / "replay-binding.json")
        check_binding(self.source, value)
        return value

    def jobs(self, config, phase):
        require(phase in ("preflight", "main"), "Known phase")
        return self.job_factory(config, phase, read_json(self.root / "replay-binding.json"))

    def execution(self, config, phase):
        records, missing = [], []
        for job in self.jobs(config, phase):
            if seal_path(self.root, phase, job["task"]["key"]).exists():
                value = verify_seal(self.root, phase, job)
                records.append(
                    dict(kind=job["kind"], interrupted=value["interrupted"], **value["result"])
                )
            else:
                missing.append(job["task"]["key"])
        return dict(
            phase=phase,
            completed=not missing,
            records=records,
            unattempted_jobs=missing,
            planned=self.design["counts"](config, phase)["workflows"],
            attempted=sum(r["attempted"] or 0 for r in records),
            neural_fits=sum(r["neural_fits"] or 0 for r in records),
            failed_points=sum(r["failed_points"] or 0 for r in records),
            unknown_counter_units=sum(r["attempted"] is None for r in records),
            serial_job_seconds=sum(r["seconds"] for r in records),
            freeze_sha256=fingerprint_file(self.root / "frozen.json").sha256,
        )

    def run(self, phase, *, workers=1):
        root = self.root
        _, config = self.check()
        require(type(workers) is int and 1 <= workers <= config.max_workers, "Worker ceiling")
        jobs = self.jobs(config, phase)
        staging = checked_gate(root) if phase == "main" else 0
        require(not (root / "STOP.json").exists(), "Retained guard requires amendment")
        with StudyLock(root):
            path = root / (phase + "-execution.json")
            if path.exists():
                try:
                    self.binding()
                    value = self.execution(config, phase)
                    require(value == read_json(path), "Immutable completed phase")
                    return {k: v for k, v in value.items() if k != "records"}
                except Exception as exc:
                    stop(root, exc)
                    raise
            started = time.monotonic()
            clock = root / "state/controllers" / phase / (uuid.uuid4().hex + ".json")
            atomic_json(clock, dict(started=started, elapsed=None))
            try:
                self.binding()
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
                                self.runner,
                                {
                                    **job,
                                    "phase_root": str(root / phase),
                                    "archive_root": str(self.source),
                                    "stop_file": str(root / "STOP.json"),
                                },
                            )
                            for job in chunk
                        ]
                        for job, future in zip(chunk, futures, strict=True):
                            seal(root, phase, job, future.result())
                        self.check()
                        resource_guard(root, config, phase, staging_bytes=staging)
                        print(
                            dict(
                                phase=phase,
                                units_completed=offset + len(chunk),
                                untouched=len(pending) - offset - len(chunk),
                            ),
                            flush=True,
                        )
                require(not (root / "STOP.json").exists(), "Retained numerical guard")
                value = self.execution(config, phase)
                if value["completed"]:
                    atomic_json(path, value)
                self.binding()
                return {k: v for k, v in value.items() if k != "records"}
            except Exception as exc:
                stop(root, exc)
                raise
            finally:
                atomic_json(
                    clock, dict(started=started, elapsed=time.monotonic() - started), replace=True
                )

    def audit(self, phase):
        from threadpoolctl import threadpool_limits

        root = self.root
        _, config = self.check()
        with StudyLock(root):
            try:
                require(not (root / "STOP.json").exists(), "Retained guard requires amendment")
                self.binding()
                execution = read_json(root / (phase + "-execution.json"))
                require(
                    execution == self.execution(config, phase) and execution["completed"],
                    "Complete immutable phase",
                )
                path = root / (phase + "-audit.json")
                if path.exists():
                    value = read_json(path)
                    require(
                        value["passed"]
                        and value["execution_sha256"]
                        == fingerprint_file(root / (phase + "-execution.json")).sha256,
                        "Matching independently audited execution",
                    )
                    return {k: v for k, v in value.items() if k != "records"}
                started, cache, records = time.monotonic(), {}, []
                with threadpool_limits(limits=1):
                    for job in self.jobs(config, phase):
                        kept = verify_seal(root, phase, job)
                        if kept["interrupted"]:
                            records.append(
                                dict(
                                    key=job["task"]["key"],
                                    interrupted=True,
                                    neural_fits=0,
                                    verified_updates=0,
                                    numerical_audit="Unavailable; bytes retained and verified",
                                )
                            )
                        else:
                            records.extend(
                                self.auditor(
                                    root / phase / job["task"]["key"], job, self.source, cache=cache
                                )
                            )
                require(
                    sum(r["neural_fits"] for r in records) == execution["neural_fits"],
                    "Known neural denominator; interrupted counts remain unknown",
                )
                value = dict(
                    passed=True,
                    phase=phase,
                    records=records,
                    seconds=time.monotonic() - started,
                    neural_fits=execution["neural_fits"],
                    verified_optimizer_updates=sum(r["verified_updates"] for r in records),
                    freeze_sha256=fingerprint_file(root / "frozen.json").sha256,
                    execution_sha256=fingerprint_file(root / (phase + "-execution.json")).sha256,
                )
                atomic_json(root / "audits" / phase / "all.json", value)
                atomic_json(path, value)
                resource_guard(
                    root, config, phase, staging_bytes=checked_gate(root) if phase == "main" else 0
                )
                self.binding()
                return {k: v for k, v in value.items() if k != "records"}
            except Exception as exc:
                stop(root, exc)
                raise
