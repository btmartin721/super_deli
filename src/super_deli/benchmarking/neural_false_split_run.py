"""Unchanged native pipeline refits on fresh or source-bound archived raw tables."""

import os
import shutil
import socket
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_calibration_config import model_for
from super_deli.benchmarking.neural_calibration_run import run_arm, run_dataset
from super_deli.benchmarking.neural_false_split_config import ARMS
from super_deli.benchmarking.neural_false_split_state import atomic_json
from super_deli.io.config import load_config, write_json


def calibration_policy(task, arm):
    seed = int(np.random.SeedSequence(task["null_seed_path"]).generate_state(1, dtype=np.uint64)[0])
    return replace(model_for(arm, seed), n_simulations=task["n_simulations"])


def run_archived(job):
    started = time.monotonic()
    t = job["task"]
    out = Path(job["phase_root"]) / t["key"]
    source = Path(job["archive_root"]) / job["source_path"]
    out.mkdir()
    write_json(t, out / "task.json")
    shutil.copytree(source / "raw", out / "raw")
    shutil.copy2(source / "generating-reference.npz", out / "generating-reference.npz")
    dataset = load_config(out / "raw/dataset.json")
    rows = []
    for arm in ARMS:
        require(not Path(job["stop_file"]).exists(), "Earlier guard prohibits dependent work")
        rows.append(run_arm(dataset, out / "raw", out / arm, calibration_policy(t, arm), arm))
        if rows[-1]["numerical_guard_failed"]:
            write_json(
                dict(reason="Actual optimizer guard", key=t["key"], arm=arm), Path(job["stop_file"])
            )
    write_json(rows, out / "fits.json")
    return dict(
        key=t["key"],
        seconds=time.monotonic() - started,
        attempted=sum(r["attempted"] for r in rows),
        neural_fits=sum(r["neural_fits"] for r in rows),
        failed_points=sum(r["failed_points"] for r in rows),
    )


def run_unit(job):
    phase = Path(job["phase_root"])
    marker = phase.parent / "state/active" / (phase.name + "_" + job["task"]["key"] + ".json")
    active = dict(
        pid=os.getpid(),
        host=socket.gethostname(),
        status="running",
        key=job["task"]["key"],
        started=time.monotonic(),
        elapsed=None,
    )
    atomic_json(marker, active)
    try:
        return run_dataset(job) if job["kind"] == "fresh" else run_archived(job)
    finally:
        atomic_json(
            marker,
            {**active, "status": "finished", "elapsed": time.monotonic() - active["started"]},
            replace=True,
        )
