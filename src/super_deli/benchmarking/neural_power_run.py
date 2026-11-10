"""Full paired native refits; archived execution composes the existing adapter."""

import os
import socket
import time
from pathlib import Path

import numpy as np

from super_deli.benchmarking.matched_data import write_dataset
from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_calibration_run import run_arm
from super_deli.benchmarking.neural_false_split_run import calibration_policy, run_archived
from super_deli.benchmarking.neural_false_split_state import atomic_json
from super_deli.benchmarking.neural_power_config import ARMS
from super_deli.benchmarking.neural_power_data import generate
from super_deli.io.config import write_json


def run_dataset(job):
    started = time.monotonic()
    task = job["task"]
    out = Path(job["phase_root"]) / task["key"]
    out.mkdir()
    write_json(task, out / "task.json")
    data = generate(task)
    np.savez_compressed(out / "generating-reference.npz", **data)
    dataset = write_dataset(data, out / "raw")
    rows = []
    for arm in ARMS:
        require(not Path(job["stop_file"]).exists(), "Earlier numerical guard")
        rows.append(run_arm(dataset, out / "raw", out / arm, calibration_policy(task, arm), arm))
        if rows[-1]["numerical_guard_failed"]:
            atomic_json(Path(job["stop_file"]), dict(reason="Actual optimizer guard", arm=arm))
    write_json(rows, out / "fits.json")
    return dict(
        key=task["key"],
        seconds=time.monotonic() - started,
        attempted=sum(r["attempted"] for r in rows),
        neural_fits=sum(r["neural_fits"] for r in rows),
        failed_points=sum(r["failed_points"] for r in rows),
    )


def run_unit(job):
    phase = Path(job["phase_root"])
    marker = phase.parent / "state/active" / (phase.name + "_" + job["task"]["key"] + ".json")
    value = dict(
        pid=os.getpid(),
        host=socket.gethostname(),
        status="running",
        key=job["task"]["key"],
        started=time.monotonic(),
        elapsed=None,
    )
    atomic_json(marker, value)
    try:
        return run_dataset(job) if job["kind"] == "fresh" else run_archived(job)
    finally:
        atomic_json(
            marker,
            {**value, "status": "finished", "elapsed": time.monotonic() - value["started"]},
            replace=True,
        )
