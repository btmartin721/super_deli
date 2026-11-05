"""Exclusive archived-coordinate evaluations; actual failures retain all evidence."""

import time
from pathlib import Path

import numpy as np

from super_deli.benchmarking.coordinate_archive import load_context
from super_deli.benchmarking.coordinate_config import CLUSTERERS, SPACES
from super_deli.benchmarking.coordinate_fit import AuditFailure, density, frozen_only, mixture
from super_deli.benchmarking.coordinate_geometry import diagnostics
from super_deli.benchmarking.separate_run import record_stop
from super_deli.io.config import write_json


def evaluate(point, reference, space, folder):
    records = []
    started = time.monotonic()
    folder.mkdir(exist_ok=False)
    np.savez_compressed(folder / "coordinates.npz", scores=point["scores"])
    for clusterer in CLUSTERERS:
        out = folder / clusterer
        out.mkdir()
        start = time.monotonic()
        data = {}
        fatal = False
        try:
            if clusterer == "hdbscan":
                record, data = density(point, space)
                geometry, geometry_arrays = diagnostics(
                    point["scores"], reference, data["linkage"], data["condensed"]
                )
                data.update(geometry_arrays)
                write_json(geometry, folder / "geometry.json")
            else:
                record, data = mixture(
                    point["scores"], "full" if clusterer == "gmm_full" else "tied"
                )
        except Exception as exc:
            fatal = True
            record = dict(
                status="failed", selected_k=None, error=str(exc), numerical_guard_failed=True
            )
        record.update(
            arm=space + "__" + clusterer,
            space=space,
            clusterer=clusterer,
            seconds=time.monotonic() - start,
            planned=1,
            attempted=1,
            neural_fits_attempted=0,
            draws_requested=0,
            numerical_guard_failed=fatal,
        )
        np.savez_compressed(out / "fit.npz", **data)
        write_json(record, out / "execution.json")
        records.append(record)
        if fatal:
            break
    write_json(
        dict(seconds=time.monotonic() - started, requested=3, attempted=len(records)),
        folder / "execution.json",
    )
    return records


def run_dataset(job):
    start = time.monotonic()
    task = job["task"]
    root = Path(job["phase_root"]) / task["key"]
    root.mkdir(exist_ok=False)
    write_json(task, root / "task.json")
    records = []
    source_error = None
    with frozen_only():
        try:
            context = load_context(job["archive_root"], task)
        except Exception as exc:
            source_error = str(exc)
            write_json(dict(error=source_error), root / "source-failure.json")
            record_stop(
                job["stop_file"],
                dict(task=task["key"], reason="Source/projection guard", error=source_error),
            )
        for space in SPACES:
            if Path(job["stop_file"]).exists():
                for clusterer in CLUSTERERS:
                    records.append(
                        dict(
                            arm=space + "__" + clusterer,
                            space=space,
                            clusterer=clusterer,
                            status="unattempted",
                            planned=1,
                            attempted=0,
                            neural_fits_attempted=0,
                        )
                    )
                continue
            rows = evaluate(context["spaces"][space], context["reference"], space, root / space)
            records.extend(rows)
            if any(r["numerical_guard_failed"] for r in rows):
                record_stop(
                    job["stop_file"],
                    dict(task=task["key"], space=space, reason="Density/numerical audit guard"),
                )
                for clusterer in CLUSTERERS[len(rows) :]:
                    records.append(
                        dict(
                            arm=space + "__" + clusterer,
                            space=space,
                            clusterer=clusterer,
                            status="unattempted",
                            planned=1,
                            attempted=0,
                            neural_fits_attempted=0,
                        )
                    )
    write_json(records, root / "fits.json")
    return dict(
        key=task["key"],
        seconds=time.monotonic() - start,
        planned=30,
        attempted=sum(r["attempted"] for r in records),
        neural_fits=0,
        failed_points=sum(r["status"] == "failed" for r in records),
        numerical_guard_failed=source_error is not None
        or any(r.get("numerical_guard_failed") for r in records),
    )


def run_replay(job):
    raise AuditFailure(
        "No additional replay jobs are planned; every baseline evaluation is a source replay"
    )
