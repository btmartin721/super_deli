"""Installed opt-in calibration on invented complete data; no empirical source."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from super_deli.benchmarking.matched_data import write_dataset
from super_deli.benchmarking.neural_calibration_config import model_for
from super_deli.io.config import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    out = parser.parse_args().output_dir
    out.mkdir()
    rng = np.random.default_rng(110101)
    write_dataset(
        dict(
            morphology=rng.normal(size=(32, 4)),
            ecology=rng.normal(size=(32, 8)),
            ecology_order=rng.permutation(32),
        ),
        out / "raw",
    )
    c = model_for("neural", 110103).to_dict()
    c["n_simulations"] = 2
    write_json(c, out / "calibration-policy.json")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "super_deli",
            "calibrate-matched-separate-gmm",
            str(out / "raw/dataset.json"),
            "--calibration-config",
            str(out / "calibration-policy.json"),
            "--output-dir",
            str(out / "calibration"),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (completed.stdout[-3000:], completed.stderr[-3000:])
    result = json.loads(completed.stdout)
    assert result["attempted_fits"] == result["successful_fits"] == 3
    assert result["inference"]["requested"] == 2
    assert result["inference"]["p_value"] is not None
    assert result["inference"]["resolution_reaches_alpha"] is False
    assert (out / "calibration/pipeline_artifacts.zip").is_file()
    print("Installed full-pipeline neural calibration and raw-draw refits passed")


if __name__ == "__main__":
    main()
