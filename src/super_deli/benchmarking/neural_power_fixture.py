"""Invented calibration archive for engineering and installation checks only."""

from pathlib import Path

from super_deli.benchmarking.neural_power_config import NeuralPowerStudy, tasks
from super_deli.benchmarking.neural_power_run import run_dataset
from super_deli.io.config import write_json


def create(source):
    source = Path(source)
    source.mkdir()
    (source / "main").mkdir()
    task = tasks(NeuralPowerStudy(fixture=True), "preflight")[0]
    for name in (
        "data_seed",
        "noise_seed",
        "permutation_seed",
        "ecology_order_seed",
        "null_seed_path",
    ):
        task[name] = [*task[name][:2], 3, *task[name][3:]]
    task["key"] = "invented_archive"
    value = run_dataset(
        dict(task=task, phase_root=str(source / "main"), stop_file=str(source / "STOP.json"))
    )
    if value["attempted"] != 6 or value["failed_points"]:
        raise ValueError("Complete invented calibration required")
    write_json(
        dict(kind="invented_power_archive_v1", paths=["main/invented_archive"]),
        source / "fixture.json",
    )
    return value
