"""Native raw-input workflows observed without replacing numerical operations."""

from pathlib import Path

import numpy as np

from super_deli.benchmarking.full_retention_audit import audit_run
from super_deli.benchmarking.full_retention_observer import observe_pipeline
from super_deli.benchmarking.multimodal_validation import observe_shape_refits
from super_deli.benchmarking.signal_loss_fit import observe_candidates
from super_deli.benchmarking.signal_loss_metrics import recovery
from super_deli.io.config import read_json, write_json
from super_deli.workflows.integrative import run_integrative


def run_native(
    dataset, model, base, directory, *, law, truth, auditor=audit_run, observer=observe_candidates
):
    directory = Path(directory)
    capture = directory.parent / (directory.name + "-captures")
    capture.mkdir()
    error = None
    shapes = []
    try:
        with observer(capture / "observed-candidates") as candidates:
            with observe_shape_refits() as shapes:
                with observe_pipeline(capture / "observed-pipeline", candidates):
                    native = run_integrative(
                        dataset, model_config=model, base_dir=base, output_dir=directory
                    )
    except AssertionError:
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    write_json(shapes, capture / "observed-shape.json")
    # Move only this run's read-only captures alongside the native evidence.
    # A failed native run still has an output directory and failure receipt.
    directory.mkdir(exist_ok=True)
    for child in capture.iterdir():
        child.rename(directory / child.name)
    capture.rmdir()
    row = dict(
        status="failed" if error else "complete",
        error=error,
        configuration=model.to_dict(),
        planned_searches=1 + model.n_bootstrap,
    )
    if error is None:
        ids = native["unit_ids"]
        lookup = dict(zip(truth["unit_ids"], truth["labels"], strict=True))
        if set(ids) != set(lookup):
            raise AssertionError("Truth/cohort mismatch")
        y = np.asarray([lookup[u] for u in ids])
        with np.load(directory / "mixture.npz", allow_pickle=False) as a:
            row.update(recovery(y, a["labels"], law))
            row.update(selected_k=native["selected_k"], labels=a["labels"].tolist())
        row["stability"] = native["bootstrap"]
    audit = auditor(directory)
    write_json(audit, directory / "independent-audit.json")
    row["audit"] = audit
    write_json(row, directory / "validation.json")
    return row


def read_case(directory):
    return read_json(Path(directory) / "validation.json")
