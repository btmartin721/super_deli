"""Dataset-level uncertainty and paired screening/recovery tradeoffs."""

from __future__ import annotations

import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from super_deli.benchmarking.grouped_false_positive import paired_decisions, rejection_rate
from super_deli.benchmarking.grouped_power_analysis import describe
from super_deli.benchmarking.local_archives import write_progress
from super_deli.benchmarking.predictive_audit import audit_observation
from super_deli.io.config import read_json, write_json
from super_deli.models.predictive_study_config import PredictiveStudyConfig


def summarize(rows, config):
    ids = [(r["cell"], r["repetition"]) for r in rows]
    expected = {(r["cell"], r["repetition"]) for r in config.plan()}
    if len(ids) != len(expected) or set(ids) != expected:
        raise ValueError("Missing, duplicate or unplanned observations")
    primary_cells = [c for c in config.cells if c.role in {"control", "alternative"}]
    family = 2 * len(primary_cells)
    cells = []
    primary = []
    comparisons = []
    for cell in config.cells:
        group = sorted([r for r in rows if r["cell"] == cell.name], key=lambda r: r["repetition"])
        event = 1 if cell.role != "alternative" else 4
        methods = {}
        for method in ("ordinary", "screened", "bic"):
            good = [r["performance"][method] for r in group if r["performance"][method] is not None]
            events = sum((r["k"] > 1 if event == 1 else r["k"] == 4) for r in good)
            is_primary = method != "bic" and cell.role in {"control", "alternative"}
            rate = rejection_rate(
                events, len(good), cell.n_datasets, family_size=family if is_primary else 1
            )
            methods[method] = {
                "available": len(good),
                "withheld": cell.n_datasets - len(good),
                "k_counts": {str(k): sum(r["k"] == k for r in good) for k in range(1, 5)},
                "ari": describe([r["ari"] for r in good]),
                "external_logdensity_gain": describe(
                    [r["external_mean_logdensity_gain"] for r in good]
                ),
                "negative_external_gain": sum(r["external_mean_logdensity_gain"] < 0 for r in good),
                "event": "K>1" if event == 1 else "K=4",
                "rate": rate,
                "primary": is_primary,
            }
            if is_primary:
                primary.append(
                    {"cell": cell.name, "method": method, "event": methods[method]["event"], **rate}
                )
        cells.append({"cell": cell.to_dict(), "methods": methods})
        if cell in primary_cells:
            decisions = {
                m: {
                    r["repetition"]: None
                    if r["performance"][m] is None
                    else (
                        r["performance"][m]["k"] > 1
                        if event == 1
                        else r["performance"][m]["k"] == 4
                    )
                    for r in group
                }
                for m in ("ordinary", "screened")
            }
            comparison = paired_decisions(
                decisions["screened"], decisions["ordinary"], error=0.05 / len(primary_cells)
            )
            comparison = {
                k.replace("known", "ordinary")
                .replace("fitted", "screened")
                .replace("rejections", "events"): v
                for k, v in comparison.items()
            }
            comparisons.append(
                {
                    "cell": cell.name,
                    "event": methods["ordinary"]["event"],
                    "family_size": len(primary_cells),
                    **comparison,
                }
            )
    return {
        "observations": len(rows),
        "planned_fits": sum(r["planned_fits"] for r in rows),
        **{
            k: sum(r[k] for r in rows)
            for k in ("attempted_fits", "successful_fits", "failed_fits", "unattempted_fits")
        },
        "screening_withheld": sum(r["selection"]["status"] != "complete" for r in rows),
        "primary_family_size": family,
        "primary_rates": primary,
        "cells": cells,
        "paired": comparisons,
        "no_audit_refits": True,
        "calibration_draws": 0,
        "stability_draws": 0,
        "scope": "Predictive-screen selection/recovery; no new p-values or species-boundary claims",
    }


def audit_study(source, *, output_dir, scratch, workers=1):
    root = Path(source).resolve()
    destination = Path(output_dir).resolve()
    if root == destination or root.is_relative_to(destination) or destination.is_relative_to(root):
        raise ValueError("Keep audit output separate from immutable study")
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four single-thread workers")
    config = PredictiveStudyConfig.from_dict(read_json(root / "configuration.json"))
    plan = config.plan()
    if (
        read_json(root / "plan.json") != plan
        or read_json(root / "summary.json")["status"] != "complete"
    ):
        raise ValueError("Study plan/completion differs")
    ledger = [json.loads(s) for s in (root / "observations.jsonl").read_text().splitlines()]
    mapping = {(r["cell"], r["repetition"]): r for r in ledger}
    if len(mapping) != len(ledger) or set(mapping) != {(r["cell"], r["repetition"]) for r in plan}:
        raise ValueError("Study ledger differs from planned observation identities")
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    with (
        ProcessPoolExecutor(max_workers=workers) as executor,
        gzip.open(destination / "points.jsonl.gz", "xt") as out,
    ):
        tasks = (
            (
                str(root),
                r,
                mapping[r["cell"], r["repetition"]],
                config.to_dict(),
                str(Path(scratch).resolve()),
            )
            for r in plan
        )
        for row in executor.map(audit_observation, tasks, chunksize=1):
            rows.append(row)
            out.write(json.dumps(row, sort_keys=True) + "\n")
            write_progress(
                {"observations_audited": len(rows), "requested": len(plan)},
                destination / "progress.json",
            )
    result = summarize(rows, config)
    original = read_json(root / "summary.json")
    if any(
        result[k] != original[k]
        for k in ("attempted_fits", "successful_fits", "failed_fits", "unattempted_fits")
    ):
        raise ValueError("Independent fit accounting differs")
    write_json(result, destination / "summary.json")
    return result
