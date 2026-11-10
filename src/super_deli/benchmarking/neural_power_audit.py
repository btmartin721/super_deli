"""Independent new raw laws; reuse unchanged raw-null, training and mixture audits."""

import numpy as np

from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_calibration_audit import audit_arm
from super_deli.benchmarking.neural_false_split_audit import archive_parity, check_policy
from super_deli.benchmarking.neural_power_config import ARMS
from super_deli.benchmarking.neural_power_data import reference
from super_deli.io.config import read_json


def audit_unit(folder, job, source, *, cache):
    task = job["task"]
    require(read_json(folder / "task.json") == task, "Exact frozen data task")
    check_policy(folder, task)
    data, expected = arrays(folder / "generating-reference.npz"), reference(task)
    require(data.keys() == expected.keys(), "Complete raw law evidence")
    for key in data:
        np.testing.assert_allclose(data[key], expected[key], atol=1e-12, rtol=1e-12)
    rows = [audit_arm(folder, task, arm, cache=cache) for arm in ARMS]
    left, right = (r.pop("paired_raw_inputs") for r in rows)
    for key in left.keys() & right.keys():
        np.testing.assert_array_equal(left[key], right[key])
    if job["kind"] == "replay":
        rows[0]["archival_arrays_checked"] = archive_parity(
            folder, source / job["source_path"], full=True
        )
    return rows
