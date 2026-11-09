"""Frozen-policy checks, independent numerical audits and complete archival parity."""

import json
from zipfile import ZipFile

import numpy as np

from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_calibration_audit import audit_dataset, zip_arrays
from super_deli.benchmarking.neural_false_split_config import ARMS
from super_deli.benchmarking.neural_false_split_run import calibration_policy
from super_deli.core.provenance import configuration_hash
from super_deli.io.config import read_json


def check_policy(folder, task):
    for arm in ARMS:
        expected = json.loads(json.dumps(calibration_policy(task, arm).to_dict()))
        out = folder / arm
        actual = read_json(out / "execution.json")
        require(
            actual["configuration"] == expected and not actual["point_only"],
            "Complete prespecified calibration policy",
        )
        cal = out / "calibration"
        if (cal / "configuration.json").exists():
            require(read_json(cal / "configuration.json") == expected, "Frozen stored policy")
        if (cal / "summary.json").exists():
            summary = read_json(cal / "summary.json")
            require(
                summary["configuration_sha256"] == configuration_hash(expected),
                "Summary bound to expected policy",
            )
            inf, evidence = summary["inference"], read_json(cal / "evidence.json")
            require(expected["alpha"] == 0.05, "Prespecified alpha")
            if summary["point"]["status"] == "complete":
                require(inf["alpha"] == expected["alpha"], "Prespecified inference alpha")
                require(
                    inf["minimum_p_value"] == 1 / (task["n_simulations"] + 1), "Exact resolution"
                )
            else:
                require(
                    inf["status"] == "point_failed" and inf["p_value"] is None,
                    "Failed observed point retains unavailable inference",
                )
            require(inf["requested"] == task["n_simulations"], "Frozen draw count")
            require(
                evidence["value"] == inf["p_value"]
                and evidence["replicates"] == task["n_simulations"],
                "Consistent typed evidence",
            )


def equal_arrays(a, b):
    require(a.keys() == b.keys(), "Identical scientific array inventory")
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def archive_parity(folder, original, *, full):
    checked = 0
    for arm in ARMS:
        left, right = folder / arm / "calibration", original / arm / "calibration"
        equal_arrays(arrays(left / "null_model.npz"), arrays(right / "null_model.npz"))
        require(
            read_json(left / "layout.json") == read_json(right / "layout.json"),
            "Original identity/feature layout",
        )
        config, old_config = (
            read_json(left / "configuration.json"),
            read_json(right / "configuration.json"),
        )
        if not full:
            require(config["random_seed"] != old_config["random_seed"], "Fresh diagnostic stream")
            config = {k: v for k, v in config.items() if k not in ("random_seed", "n_simulations")}
            old_config = {
                k: v for k, v in old_config.items() if k not in ("random_seed", "n_simulations")
            }
        require(config == old_config, "Unchanged archived analysis settings")
        a, b = read_json(left / "observed.json"), read_json(right / "observed.json")
        records = [(a, b)]
        if full:
            xs, ys = (
                read_json(left / "null_replicates.json"),
                read_json(right / "null_replicates.json"),
            )
            require(len(xs) == len(ys), "Replay draw count")
            records.extend(zip(xs, ys, strict=True))
            require(
                read_json(left / "summary.json")["inference"]
                == read_json(right / "summary.json")["inference"],
                "Exact original inference",
            )
        with (
            ZipFile(left / "pipeline_artifacts.zip") as x,
            ZipFile(right / "pipeline_artifacts.zip") as y,
        ):
            for a, b in records:
                for name in (
                    "status",
                    "statistic",
                    "selected_k",
                    "analysis_seed",
                    "simulation_seed",
                    "replicate",
                ):
                    require(
                        a.get(name) == b.get(name), "Replay or observed numerical result: " + name
                    )
                lp, rp = a["archive_prefix"] + "/run/", b["archive_prefix"] + "/run/"
                for name in ("input.npz", "point.npz", "failed_point.npz", "fusion.npz"):
                    require(
                        (lp + name in x.namelist()) == (rp + name in y.namelist()),
                        "Retained scientific replay artifacts",
                    )
                    if lp + name in x.namelist():
                        equal_arrays(zip_arrays(x, lp + name), zip_arrays(y, rp + name))
                        checked += 1
                if lp + "candidates.json" in x.namelist():
                    require(
                        json.loads(x.read(lp + "candidates.json"))
                        == json.loads(y.read(rp + "candidates.json")),
                        "Exact candidate results",
                    )
    return checked


def audit_unit(folder, job, source, *, cache):
    task = job["task"]
    require(read_json(folder / "task.json") == task, "Frozen raw-data task")
    check_policy(folder, task)
    rows = audit_dataset(folder, task, cache=cache)
    if job["kind"] != "fresh":
        checked = archive_parity(folder, source / job["source_path"], full=job["kind"] == "replay")
        rows[0]["archival_arrays_checked"] = checked
    return rows
