"""Raw-input calibration runs with live read-only neural checks and paired draws."""

import time
from pathlib import Path

import numpy as np

from super_deli.benchmarking.matched_data import write_dataset
from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_calibration_config import ARMS, model_for
from super_deli.benchmarking.neural_calibration_data import generate
from super_deli.benchmarking.neural_calibration_points import pca_calibration, pca_runner
from super_deli.benchmarking.separate_gmm_observer import retain_points
from super_deli.benchmarking.view_weight_observer import retain_training
from super_deli.io.config import load_config, read_json, write_json
from super_deli.workflows.neural_calibration import fit_input, run_neural_calibration


def run_arm(dataset, raw, out, config, arm, *, point_only=False):
    from super_deli.models import tabular_neural

    started = time.monotonic()
    out.mkdir()
    guard = {}
    events = []
    error = None
    result = None
    try:
        if arm == "neural":
            with retain_training(
                out, paired=False, module=tabular_neural, guard_state=guard, record_times=True
            ) as events:
                with retain_points(out):
                    result = (
                        fit_input(
                            dataset, model_config=config.model, base_dir=raw, output_dir=out / "run"
                        )
                        if point_only
                        else run_neural_calibration(
                            dataset, config=config, base_dir=raw, output_dir=out / "calibration"
                        )
                    )
        else:
            result = (
                pca_runner(
                    dataset,
                    model_config=config.model,
                    base_dir=raw,
                    output_dir=out / "run",
                    evidence_dir=out / "head-evidence" / "0",
                )
                if point_only
                else pca_calibration(
                    dataset, config=config, base_dir=raw, output_dir=out / "calibration"
                )
            )
    except Exception as exc:
        error = str(exc)
    if point_only:
        attempts = 1
        failed = int(error is not None)
    elif result:
        attempts = result["attempted_fits"]
        failed = result["failed_fits"]
    else:
        failure = (
            read_json(out / "calibration/failure.json")
            if (out / "calibration/failure.json").exists()
            else {}
        )
        attempts = failure.get("attempted_fits", 0)
        failed = failure.get("failed_fits", 0)
    row = dict(
        arm=arm,
        status="available" if result is not None else "failed",
        error=error,
        configuration=config.to_dict(),
        point_only=point_only,
        attempted=attempts,
        failed_points=failed,
        neural_fits=len(events),
        seconds=time.monotonic() - started,
        numerical_guard_failed=bool(guard.get("failed")),
    )
    write_json(row, out / "execution.json")
    return row


def run_dataset(job):
    from dataclasses import replace

    started = time.monotonic()
    t = job["task"]
    out = Path(job["phase_root"]) / t["key"]
    out.mkdir()
    write_json(t, out / "task.json")
    data = generate(t)
    np.savez_compressed(out / "generating-reference.npz", **data)
    ds = write_dataset(data, out / "raw")
    seed = int(np.random.SeedSequence(t["null_seed_path"]).generate_state(1, dtype=np.uint64)[0])
    rows = []
    for arm in ARMS:
        require(not Path(job["stop_file"]).exists(), "Earlier guard prohibits dependent work")
        config = replace(model_for(arm, seed), n_simulations=t["n_simulations"])
        row = run_arm(ds, out / "raw", out / arm, config, arm)
        rows.append(row)
        if row["numerical_guard_failed"]:
            write_json(
                dict(reason="Actual optimizer guard", key=t["key"], arm=arm), Path(job["stop_file"])
            )
    write_json(rows, out / "fits.json")
    return dict(
        key=t["key"],
        seconds=time.monotonic() - started,
        attempted=sum(r["attempted"] for r in rows),
        neural_fits=sum(r["neural_fits"] for r in rows),
        failed_points=sum(r["failed_points"] for r in rows),
    )


def replay_equal(out, source, entry):
    old = source / entry["path"]
    for arm, previous in (("neural", "separate_0"), ("pca4", "pca4")):
        a = arrays(out / arm / "run/point.npz")
        reference = (
            arrays(old / (arm + ".npz"))
            if entry["fixture"]
            else arrays(old / previous / "point.npz")
        )
        for key, value in reference.items():
            if key not in ("labels", "strength") or arm == "neural":
                np.testing.assert_array_equal(a[key], value)
        if not entry["fixture"]:
            p = old / previous / "evaluations/point/gmm_shared/fit.npz"
            head = arrays(p)
            np.testing.assert_array_equal(a["labels"], head["labels"])
            if arm == "neural":
                np.testing.assert_array_equal(a["membership"], head["membership"])


def run_replay(job):
    started = time.monotonic()
    e = job["entry"]
    source = Path(job["archive_root"])
    out = Path(job["phase_root"]) / e["key"]
    out.mkdir()
    write_json(e, out / "replay.json")
    raw = source / e["path"] / "raw"
    ds = load_config(raw / "dataset.json")
    rows = [run_arm(ds, raw, out / a, model_for(a, 0), a, point_only=True) for a in ARMS]
    write_json(rows, out / "fits.json")
    replay_equal(out, source, e)
    return dict(
        key=e["key"],
        seconds=time.monotonic() - started,
        attempted=sum(r["attempted"] for r in rows),
        neural_fits=sum(r["neural_fits"] for r in rows),
        failed_points=sum(r["failed_points"] for r in rows),
        exact_numeric_replay=True,
    )
