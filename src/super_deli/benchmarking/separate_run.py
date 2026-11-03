"""Fixed family dispatch with live numerical evidence and bounded failure propagation."""

import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from super_deli.benchmarking.matched_data import write_dataset
from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_moons_coordinates import safe_path
from super_deli.benchmarking.separate_config import ARMS, NATIVE, model_for
from super_deli.benchmarking.view_weight_data import generate
from super_deli.benchmarking.view_weight_observer import retain_density, retain_training
from super_deli.io.config import load_config, read_json, write_json


def record_stop(path, record):
    try:
        write_json(record, path)
    except FileExistsError:
        pass


def run_arm(task, data_dir, canonical_dir, out, arm):
    if arm not in NATIVE and arm != "pca4":
        from super_deli.benchmarking.view_weight_run import run_arm as run_reference

        return run_reference(task, data_dir, canonical_dir, out, arm, record_times=True)
    from super_deli.benchmarking import separate_controls as controls
    from super_deli.models import matched_neural, separate_neural, tabular_neural
    from super_deli.workflows.matched_neural import run_matched_neural
    from super_deli.workflows.separate_neural import run_separate_neural

    seed = int(np.random.SeedSequence(task["subsampling_path"]).generate_state(1)[0])
    config = model_for(arm, seed, task["draws"])
    is_separate = arm.startswith("separate_")
    is_joint2 = arm.startswith("joint2_")
    neural = arm != "pca4"
    workflow = (
        run_separate_neural
        if is_separate
        else run_matched_neural
        if is_joint2
        else controls.run_control
    )
    training_module = tabular_neural if is_separate else matched_neural if is_joint2 else controls
    density_module = separate_neural if is_separate else training_module
    factory = controls.four_coordinate_factory() if arm.startswith("joint4_") else nullcontext()
    events = []
    guard = {}
    start = time.monotonic()
    try:
        with factory:
            with (
                retain_training(
                    out, paired=False, module=training_module, guard_state=guard, record_times=True
                )
                if neural
                else nullcontext([])
            ) as events:
                with retain_density(
                    out, events, paired=False, module=density_module, guard_state=guard
                ):
                    summary = workflow(
                        load_config(data_dir / "dataset.json"),
                        model_config=config,
                        base_dir=data_dir,
                        output_dir=out,
                    )
        result = {"status": "available", "selected_k": summary["selected_k"]}
    except Exception as exc:
        result = {"status": "failed", "selected_k": None, "error": str(exc)}
    subs = read_json(out / "subsamples.json") if (out / "subsamples.json").exists() else []
    result.update(
        arm=arm,
        configuration=config.to_dict(),
        draws_requested=config.n_subsamples,
        draws_attempted=len(subs),
        planned=1 + config.n_subsamples,
        attempted=1 + len(subs),
        neural_fits_attempted=len(events),
        seconds=time.monotonic() - start,
        numerical_guard_failed=bool(guard.get("failed")),
    )
    write_json(result, out / "execution.json")
    return result


def run_dataset(job):
    start = time.monotonic()
    task = job["task"]
    out = Path(job["phase_root"]) / task["key"]
    out.mkdir(exist_ok=False)
    write_json(task, out / "task.json")
    data = generate(task)
    np.savez_compressed(out / "generating-reference.npz", **data)
    write_dataset(data, out / "raw")
    write_dataset(data, out / "canonical", canonical=True)
    records = []
    for arm in ARMS:
        if Path(job["stop_file"]).exists():
            records.append(
                {
                    "arm": arm,
                    "status": "unattempted",
                    "reason": "Earlier phase guard failed",
                    "planned": 1 + task["draws"] if arm in NATIVE else 1,
                    "attempted": 0,
                    "neural_fits_attempted": 0,
                }
            )
            continue
        row = run_arm(task, out / "raw", out / "canonical", out / arm, arm)
        records.append(row)
        if row["numerical_guard_failed"]:
            record_stop(
                job["stop_file"],
                {"dataset": task["key"], "arm": arm, "reason": "Numerical audit guard"},
            )
    write_json(records, out / "fits.json")
    return {
        "key": task["key"],
        "seconds": time.monotonic() - start,
        "planned": sum(r["planned"] for r in records),
        "attempted": sum(r["attempted"] for r in records),
        "neural_fits": sum(r["neural_fits_attempted"] for r in records),
        "failed_points": sum(r["status"] == "failed" for r in records),
        "numerical_guard_failed": any(r.get("numerical_guard_failed") for r in records),
    }


def replay_equal(original, folder, entry):
    expected, actual = arrays(original / entry["arm"] / "point.npz"), arrays(folder / "point.npz")
    for key, value in expected.items():
        np.testing.assert_array_equal(actual[key], value)
    old_path = original / entry["arm"] / "optimizer-checks.npz"
    initial = arrays(folder / "optimizer-checks.npz")
    old = arrays(old_path if old_path.exists() else original / entry["arm"] / "initial-weights.npz")
    for key, value in old.items():
        target = key if old_path.exists() else key.replace("fit_0_", "fit_0_initial_", 1)
        if target.startswith("fit_0_initial_"):
            np.testing.assert_array_equal(initial[target], value)


def run_replay(job):
    entry = job["entry"]
    source = safe_path(job["archive_root"], entry["path"])
    out = Path(job["phase_root"]) / entry["key"]
    out.mkdir(exist_ok=False)
    write_json(entry, out / "replay.json")
    task = {**entry["task"], "draws": 0}
    arm = "joint2_" + str(entry["seed"])
    start = time.monotonic()
    row = run_arm(task, source / "raw", source / "canonical", out / arm, arm)
    try:
        if row["status"] == "available":
            replay_equal(source, out / arm, entry)
            row["exact_numeric_replay"] = True
        require(not row["numerical_guard_failed"], "Legacy optimizer audit rejected")
    except (AssertionError, ValueError) as exc:
        row.update(status="failed", error=str(exc), numerical_guard_failed=True)
        record_stop(job["stop_file"], {"replay": entry["key"], "reason": str(exc)})
    write_json([row], out / "fits.json")
    return {
        "key": entry["key"],
        "seconds": time.monotonic() - start,
        "planned": 1,
        "attempted": 1,
        "neural_fits": row["neural_fits_attempted"],
        "failed_points": row["status"] != "available",
        "numerical_guard_failed": row["numerical_guard_failed"],
    }
