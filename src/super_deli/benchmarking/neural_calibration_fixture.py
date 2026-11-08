"""Small invented replay source for the complete CLI lifecycle unit test."""

from pathlib import Path

import numpy as np

from super_deli.benchmarking.matched_data import write_dataset
from super_deli.benchmarking.neural_analysis import arrays
from super_deli.benchmarking.neural_calibration_config import (
    NeuralCalibrationStudy,
    model_for,
    tasks,
)
from super_deli.benchmarking.neural_calibration_data import generate
from super_deli.benchmarking.neural_calibration_run import run_arm
from super_deli.io.config import write_json


def create(source):
    source = Path(source)
    source.mkdir()
    folder = source / "invented"
    folder.mkdir()
    task = tasks(NeuralCalibrationStudy(fixture=True), "preflight")[0]
    ds = write_dataset(generate(task), folder / "raw")
    for arm in ("neural", "pca4"):
        row = run_arm(ds, folder / "raw", folder / arm, model_for(arm, 0), arm, point_only=True)
        if row["status"] != "available":
            raise ValueError(row["error"])
        np.savez_compressed(folder / (arm + ".npz"), **arrays(folder / arm / "run/point.npz"))
    write_json(
        dict(
            kind="invented_neural_calibration_source",
            replays=[dict(key="replay_invented", path="invented", fixture=True)],
        ),
        source / "fixture.json",
    )
