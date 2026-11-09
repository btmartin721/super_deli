"""Small invented archived calibration for installation/engineering checks only."""

from pathlib import Path

from super_deli.benchmarking.neural_calibration_run import run_dataset
from super_deli.benchmarking.neural_false_split_config import make_task
from super_deli.io.config import write_json


def create(source):
    source = Path(source)
    source.mkdir()
    (source / "main").mkdir()
    task = make_task(3, 0, 0, n=32, draws=2, key="invented_archive")
    result = run_dataset(
        dict(task=task, phase_root=str(source / "main"), stop_file=str(source / "STOP.json"))
    )
    if result["attempted"] != 6 or result["failed_points"]:
        raise ValueError("Invented archive must have complete calibration fits")
    write_json(
        dict(kind="invented_false_split_archive_v1", paths=["main/invented_archive"]),
        source / "fixture.json",
    )
    return result
