"""Native point/subsample workflows with live independent optimizer evidence."""

import time
from pathlib import Path

import numpy as np

from super_deli.benchmarking.matched_data import write_dataset
from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.unknown_k_run import retain_gmm_candidates
from super_deli.benchmarking.view_weight_config import ARMS, JOINT, model_for
from super_deli.benchmarking.view_weight_data import generate
from super_deli.benchmarking.view_weight_observer import retain_density, retain_training
from super_deli.io.config import load_config, read_json, write_json


def run_arm(task, data_dir, canonical_dir, out, arm, *, legacy_config=None):
    from super_deli.workflows.integrative import run_integrative
    from super_deli.workflows.matched_neural import run_matched_neural
    from super_deli.workflows.tabular_neural import run_tabular_neural

    paired, neural = arm in JOINT, arm != "integrative_gmm"
    seed = int(np.random.SeedSequence(task["subsampling_path"]).generate_state(1)[0])
    draws = task["draws"] if paired else 0
    config = legacy_config or model_for(arm, seed, draws)
    start, events = time.monotonic(), []
    guard_state = {}
    try:
        if neural:
            with (
                retain_training(out, paired=paired, guard_state=guard_state) as events,
                retain_density(out, events, paired=paired, guard_state=guard_state),
            ):
                if paired:
                    summary = run_matched_neural(
                        load_config(data_dir / "dataset.json"),
                        model_config=config,
                        base_dir=data_dir,
                        output_dir=out,
                    )
                else:
                    summary = run_tabular_neural(
                        load_config(canonical_dir / "dataset.json"),
                        model_config=config,
                        base_dir=canonical_dir,
                        output_dir=out,
                        modality_name=arm.split("_")[0],
                    )
        else:
            with retain_gmm_candidates(out):
                summary = run_integrative(
                    load_config(canonical_dir / "dataset.json"),
                    model_config=config,
                    base_dir=canonical_dir,
                    output_dir=out,
                )
        result = {"status": "available", "selected_k": summary["selected_k"]}
    except Exception as exc:
        result = {"status": "failed", "selected_k": None, "error": str(exc)}
    subs = read_json(out / "subsamples.json") if (out / "subsamples.json").exists() else []
    result.update(
        arm=arm,
        configuration=config.to_dict(),
        draws_requested=draws,
        draws_attempted=len(subs),
        planned=1 + draws,
        attempted=1 + len(subs),
        neural_fits_attempted=len(events),
        seconds=time.monotonic() - start,
        numerical_guard_failed=bool(guard_state.get("failed")),
    )
    write_json(result, out / "execution.json")
    return result


def run_dataset(job):
    start = time.monotonic()
    out = Path(job["phase_root"]) / job["task"]["key"]
    out.mkdir(exist_ok=False)
    task = job["task"]
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
                    "planned": 1 + task["draws"] if arm in JOINT else 1,
                    "attempted": 0,
                    "neural_fits_attempted": 0,
                }
            )
            continue
        row = run_arm(task, out / "raw", out / "canonical", out / arm, arm)
        records.append(row)
        if row["numerical_guard_failed"]:
            try:
                write_json(
                    {"dataset": task["key"], "arm": arm, "reason": "Numerical audit guard"},
                    job["stop_file"],
                )
            except FileExistsError:
                pass
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


def run_replay(job):
    from super_deli.benchmarking.neural_analysis import arrays
    from super_deli.benchmarking.neural_moons_coordinates import safe_path

    entry = job["entry"]
    source = safe_path(job["archive_root"], entry["path"])
    out = Path(job["phase_root"]) / entry["key"]
    out.mkdir(exist_ok=False)
    write_json(entry, out / "replay.json")
    task = {**entry["task"], "draws": 0}
    arm = "joint_equal_" + str(entry["seed"])
    start = time.monotonic()
    row = run_arm(task, source / "raw", source / "canonical", out / arm, arm)
    try:
        if row["status"] == "available":
            actual = arrays(out / arm / "point.npz")
            expected = arrays(source / entry["arm"] / "point.npz")
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])
            initial = arrays(out / arm / "optimizer-checks.npz")
            old_initial = arrays(source / entry["arm"] / "initial-weights.npz")
            for key, value in old_initial.items():
                if key.startswith("fit_0_"):
                    np.testing.assert_array_equal(
                        initial["fit_0_initial_" + key.removeprefix("fit_0_")], value
                    )
            row["exact_numeric_replay"] = True
        require(not row["numerical_guard_failed"], "Legacy optimizer audit rejected")
    except (AssertionError, ValueError) as exc:
        row.update(status="failed", error=str(exc), numerical_guard_failed=True)
        try:
            write_json({"replay": entry["key"], "reason": str(exc)}, job["stop_file"])
        except FileExistsError:
            pass
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
