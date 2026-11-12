"""Whole-observation diagnostic worker; source coordinates stay frozen."""

import os
import socket
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from super_deli.benchmarking.coordinate_fit import frozen_only
from super_deli.benchmarking.neural_analysis import arrays
from super_deli.benchmarking.neural_false_split_state import atomic_json
from super_deli.benchmarking.weak_signal_config import ARMS
from super_deli.benchmarking.weak_signal_fit import replay, truth_fit
from super_deli.benchmarking.weak_signal_reference import geometry, laws, reference_result
from super_deli.io.config import read_json


def calculate(job):
    task = job["task"]
    root = Path(job["phase_root"])
    source = root.parent / "inputs" / task["key"]
    out = root / task["key"]
    out.mkdir()
    data = arrays(source / "reference.npz")
    truth = data["truth"]
    points = {a: arrays(source / (a + "-point.npz")) for a in ARMS}
    coordinates = dict(
        latent=np.column_stack((data["latent_morphology"], data["latent_ecology"])),
        raw=np.column_stack((data["morphology"], data["ecology"])),
        pca4=points["pca4"]["scores"],
        neural=points["neural"]["scores"],
    )
    distributions = laws(task, points["pca4"])
    references = {}
    for name, (mean, covariance) in distributions.items():
        references[name], saved = reference_result(coordinates[name], truth, mean, covariance)
        np.savez_compressed(out / (name + "-reference.npz"), **saved)
    rows, attempts, failed = {}, 0, 0
    for arm in ARMS:
        row = {}
        if task["diagnostic_replay"]:
            old = read_json(source / (arm + "-head.json"))
            meta, saved = replay(coordinates[arm], old, arrays(source / (arm + "-head.npz")))
            atomic_json(out / (arm + "-replay.json"), meta)
            np.savez_compressed(out / (arm + "-replay.npz"), **saved)
            row["replay"] = True
            attempts += len(meta["candidate_attempts"])
        if task["truth_fits"]:
            row["truth_fit"] = truth_fit(coordinates[arm], truth, out / (arm + "-truth"))
            attempts += 1
            failed += row["truth_fit"]["status"] != "complete"
        rows[arm] = row
    atomic_json(
        out / "analysis.json",
        dict(
            task=task,
            references=references,
            geometry={k: geometry(x, data) for k, x in coordinates.items()},
            fits=rows,
        ),
    )
    return dict(key=task["key"], attempted=attempts, failed_points=failed, neural_fits=0)


def run_unit(job):
    started = time.monotonic()
    root = Path(job["phase_root"])
    marker = root.parent / "state/active" / (root.name + "_" + job["task"]["key"] + ".json")
    value = dict(
        pid=os.getpid(),
        host=socket.gethostname(),
        status="running",
        key=job["task"]["key"],
        started=started,
        elapsed=None,
    )
    atomic_json(marker, value)
    try:
        with frozen_only(), threadpool_limits(limits=1):
            result = calculate(job)
        return dict(**result, seconds=time.monotonic() - started)
    finally:
        atomic_json(
            marker,
            {**value, "status": "finished", "elapsed": time.monotonic() - started},
            replace=True,
        )
