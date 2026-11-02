"""Whole-table workers; truth is saved separately and never passed to selectors."""

import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from super_deli.benchmarking.gaussian_graph_fit import (
    assign,
    fit_gaussian,
    legacy_selection,
    observed_graph,
    reference_profiles,
    tail_decision,
)
from super_deli.benchmarking.neural_analysis import read_raw
from super_deli.benchmarking.neural_study import write_dataset
from super_deli.benchmarking.unknown_k_data import generate
from super_deli.benchmarking.unknown_k_run import run_arm
from super_deli.io.config import write_json


def save_point(out, record, arrays):
    out.mkdir()
    np.savez_compressed(out / "point.npz", **arrays)
    write_json(record, out / "summary.json")


def run_table(task, condition, dataset, raw, out):
    start = time.monotonic()
    out.mkdir()
    ids, x = read_raw(raw)
    records = []
    with threadpool_limits(limits=1):
        try:
            fitted = fit_gaussian(x)
            np.savez_compressed(out / "gaussian.npz", **fitted)
            ledger, profiles = reference_profiles(task, condition, fitted, ids)
            write_json(ledger, out / "draw-ledger.json")
            np.savez_compressed(out / "reference-spectra.npz", **profiles)
            null_error = None
        except Exception as exc:
            null_error = str(exc)
            write_json({"status": "failed", "error": null_error}, out / "null-fit-failure.json")
        for neighbors in (5, 15):
            try:
                affinity, graph_arrays, info = observed_graph(x, ids, neighbors)
                selection, gaps = legacy_selection(graph_arrays["eigenvalues"])
                result, evidence = assign(affinity, graph_arrays, info, selection, neighbors)
                evidence["gaps"] = gaps
                arm = f"spectral_{neighbors}"
                save_point(out / arm, result, evidence)
                records.append({"arm": arm, **result})
                if null_error:
                    choice = {
                        "status": "withheld",
                        "selected_k": None,
                        "proposed_k": None,
                        "p_reference": None,
                        "reason": "unresolved_gaussian_fit",
                        "error": null_error,
                    }
                    additional = {}
                else:
                    choice, additional = tail_decision(
                        np.diff(graph_arrays["eigenvalues"][1:7]),
                        np.diff(profiles[f"n{neighbors}_pilot_spectra"][:, 1:7], axis=1),
                        np.diff(profiles[f"n{neighbors}_reference_spectra"][:, 1:7], axis=1),
                    )
                result, evidence = assign(affinity, graph_arrays, info, choice, neighbors)
                evidence.update(additional)
                arm = f"reference_{neighbors}"
                save_point(out / arm, result, evidence)
                records.append({"arm": arm, **result})
            except Exception as exc:
                # Preserve any completed sibling result. Never replace a failed graph.
                for family in ("spectral", "reference"):
                    arm = f"{family}_{neighbors}"
                    if not any(r["arm"] == arm for r in records):
                        result = {
                            "status": "failed",
                            "selected_k": None,
                            "reason": "observed_graph_failure",
                            "error": str(exc),
                        }
                        save_point(out / arm, result, {})
                        records.append({"arm": arm, **result})
        for arm in ("gmm", "density"):
            records.append(run_arm(dataset, raw, out / arm, arm, "point"))
    write_json(records, out / "fits.json")
    result = {"condition": condition, "seconds": time.monotonic() - start, "records": records}
    write_json(result, out / "execution.json")
    return result


def run_dataset(args):
    root, task = args
    out = Path(root) / task["key"]
    out.mkdir()
    write_json(task, out / "task.json")
    source = generate(task)
    np.savez_compressed(out / "generating-reference.npz", **source)
    results = []
    for condition in ("clean", "noisy"):
        parent = out / condition
        parent.mkdir()
        raw = parent / "raw"
        dataset = write_dataset(source[condition], raw)
        result = run_table(task, condition, dataset, raw, parent / "point")
        results.append({"kind": "point", **result})
        if condition in task["replay_conditions"]:
            result = run_table(task, condition, dataset, raw, parent / "replay")
            results.append({"kind": "replay", **result})
    write_json(results, out / "tables.json")
    return {"key": task["key"], "n": task["n"], "tables": results}
