"""Read-only source binding and lossless observed-point extraction."""

import json
from importlib.metadata import version
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from super_deli.benchmarking.matched_study import inventory
from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_calibration_audit import zip_arrays
from super_deli.benchmarking.neural_false_split_state import atomic_json, verify_seal
from super_deli.benchmarking.view_weight_archive import check_binding
from super_deli.benchmarking.weak_signal_config import ARMS, NUMERICAL, SOURCE_HEAD
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json


def selected(source, fixture):
    if fixture:
        marker = read_json(source / "fixture.json")
        require(marker["kind"] == "invented_weak_signal_source_v1", "Invented fixture kind")
        return marker["entries"]
    result = []
    for file in sorted((source / "main").glob("*/task.json")):
        task = read_json(file)
        if task["case"] in (
            "four_complementary",
            "gaussian_independent",
        ) and not file.parent.name.startswith("archived_"):
            result.append(dict(path=str(file.parent.relative_to(source)), task=task))
    require(len(result) == 80, "Exactly 80 original fresh observations")
    cells = {}
    for entry in result:
        task = entry["task"]
        key = (task["case"], task["n"], task["signal_multiplier"])
        cells.setdefault(key, []).append(task["replicate"])
    expected = {
        (c, n, s)
        for n in (64, 256)
        for c, strengths in (
            ("four_complementary", (0.5, 1.0, 1.5)),
            ("gaussian_independent", (1.0,)),
        )
        for s in strengths
    }
    require(
        set(cells) == expected and all(sorted(v) == list(range(10)) for v in cells.values()),
        "All eight frozen cells",
    )
    return result


def load_unit(source, entry, fixture):
    folder = source / entry["path"]
    data = arrays(folder / "generating-reference.npz")
    points, heads, parameters = {}, {}, {}
    for arm in ARMS:
        if fixture:
            points[arm] = arrays(folder / (arm + "-point.npz"))
            head = folder / arm
        else:
            with ZipFile(folder / arm / "calibration/pipeline_artifacts.zip") as z:
                points[arm] = zip_arrays(z, "observed/run/point.npz")
                x = zip_arrays(z, "observed/run/input.npz")["values"]
                np.testing.assert_array_equal(
                    x, np.column_stack((data["morphology"], data["ecology"]))
                )
                alignment = json.loads(z.read("observed/run/alignment.json"))
                require(
                    alignment["unit_ids"] == [f"unit_{i:04d}" for i in range(len(x))],
                    "Exact matched specimen identities",
                )
            head = (
                folder
                / arm
                / (
                    "evaluations/point/gmm_shared"
                    if arm == "neural"
                    else "calibration/head-evidence/0"
                )
            )
        heads[arm] = read_json(head / "execution.json")
        parameters[arm] = arrays(head / "fit.npz")
    return data, points, heads, parameters


def bind(source, root, *, fixture=False):
    source, root = Path(source).resolve(), Path(root).resolve()
    require(
        source != root and not source.is_relative_to(root) and not root.is_relative_to(source),
        "Separate source/output roots",
    )
    entries = selected(source, fixture)
    files = {}
    if not fixture:
        completed = read_json(source / "completion.json")
        require(
            completed["completed"]
            and completed["head_sha"] == SOURCE_HEAD
            and completed["ci_jobs"] == 6,
            "Completed exact source head",
        )
        old = read_json(source / "frozen.json")
        for name in NUMERICAL:
            require(version(name) == old["software"][name], "Archived numerical version: " + name)
        for name in (
            "completion.json",
            "frozen.json",
            "frozen-sources.tar.gz",
            "approved-plan.md",
            "main-audit.json",
            "main-execution.json",
            "verification-evidence/ci-verification.json",
            "amendments/runtime-repair/approval.json",
            "amendments/runtime-repair/acceptance.json",
            "amendments/runtime-repair/retired-STOP.json",
        ):
            files[name] = fingerprint_file(source / name).sha256
        ci = read_json(source / "verification-evidence/ci-verification.json")
        require(
            ci["head_sha"] == SOURCE_HEAD
            and ci["conclusion"] == "success"
            and len(ci["jobs"]) == 6,
            "Six exact-head source CI receipts",
        )
        for entry in entries:
            kept = verify_seal(source, "main", dict(kind="fresh", task=entry["task"]))
            require(not kept["interrupted"], "Complete original observation")
            files.update({entry["path"] + "/" + k: v for k, v in kept["inventory"].items()})
            name = "ledger/main/" + entry["task"]["key"] + ".json"
            files[name] = fingerprint_file(source / name).sha256
    else:
        files = inventory(source)
    root.mkdir(exist_ok=True)
    target = root / "inputs"
    target.mkdir()
    for entry in entries:
        folder = target / entry["task"]["key"]
        folder.mkdir()
        data, points, heads, parameters = load_unit(source, entry, fixture)
        atomic_json(folder / "task.json", entry["task"])
        np.savez_compressed(folder / "reference.npz", **data)
        for arm in ARMS:
            np.savez_compressed(folder / (arm + "-point.npz"), **points[arm])
            np.savez_compressed(folder / (arm + "-head.npz"), **parameters[arm])
            atomic_json(folder / (arm + "-head.json"), heads[arm])
    value = dict(
        passed=True,
        fixture=fixture,
        entries=entries,
        replays=[],
        source_sha256=files,
        extracted_sha256=inventory(target),
    )
    atomic_json(root / "replay-binding.json", value)
    check_binding(source, value)
    return dict(passed=True, observations=len(entries), bound_files=len(files))


def check_inputs(root):
    value = read_json(root / "replay-binding.json")
    require(inventory(root / "inputs") == value["extracted_sha256"], "Immutable source extracts")
