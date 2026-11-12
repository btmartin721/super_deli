"""Invented installed-package lifecycle and both source-bound abstention policies."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from super_deli.benchmarking.abstention_fixture import smoke
from super_deli.io.config import load_config
from super_deli.models.reference_abstention_config import ReferenceAbstentionConfig
from super_deli.workflows.reference_abstention import (
    calibrate_reference_abstention,
    load_reference_abstention,
    predict_reference_abstention,
)


def run(output):
    result = smoke(output, workers=2)
    point = output / "study/main/linear_standard_n32_0000"
    reference = point / "residual_0/run"
    for policy in ("distance", "probability"):
        calibration = output / ("api-calibration-" + policy)
        calibrate_reference_abstention(
            load_config(point / "raw/dataset.json"),
            load_config(point / "calibration/raw/dataset.json"),
            reference_run=reference,
            calibration_labels=point / "calibration/references.csv",
            abstention_config=ReferenceAbstentionConfig(True, policy=policy),
            training_base_dir=point / "raw",
            calibration_base_dir=point / "calibration/raw",
            output_dir=calibration,
        )
        destination = output / ("api-predictions-" + policy)
        predict_reference_abstention(
            load_config(point / "iid/raw/dataset.json"),
            reference_run=reference,
            calibration_run=calibration,
            base_dir=point / "iid/raw",
            output_dir=destination,
        )
        model, _, _, _ = load_reference_abstention(calibration, reference_run=reference)
        with np.load(point / "iid/generating-reference.npz", allow_pickle=False) as d:
            x = np.column_stack((d["morphology"], d["ecology"]))
        expected = model.predict_matrix(x)["probabilities"]
        with (destination / "reference-assignments.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        np.testing.assert_array_equal(
            np.array([[float(r[f"class_{j}_probability"]) for j in range(4)] for r in rows]),
            expected,
        )
    return dict(
        passed=True,
        classifier_trainings=8,
        neural_trainings=6,
        verified_optimizer_updates=3000,
        empirical_analysis=False,
        result=result,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), sort_keys=True))
