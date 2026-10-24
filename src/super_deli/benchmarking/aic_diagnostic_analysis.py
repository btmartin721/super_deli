"""Independent-dataset component-selection rates and paired mechanism summaries."""

from __future__ import annotations

import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from super_deli.benchmarking.aic_diagnostic_audit import audit_observation
from super_deli.benchmarking.grouped_false_positive import paired_decisions, rejection_rate
from super_deli.benchmarking.grouped_power_analysis import describe
from super_deli.benchmarking.local_archives import write_progress
from super_deli.io.config import read_json, write_json
from super_deli.models.aic_diagnostic_config import ARMS, AICDiagnosticConfig


def summarize(rows, config):
    expected = {
        (r["cell"], r["repetition"], arm)
        for r in config.plan()
        for arm in (("scores",) if r["case"] == "gaussian_scores" else ARMS)
    }
    ids = [(r["cell"], r["repetition"], r["arm"]) for r in rows]
    if set(ids) != expected or len(ids) != len(expected):
        raise ValueError("Missing, duplicate or unplanned observations/arms")
    family = len(config.cases) * len(config.sizes) * len(ARMS) + len(config.sizes)
    cells = []
    for cell, arm in sorted({(r["cell"], r["arm"]) for r in rows}):
        group = [r for r in rows if (r["cell"], r["arm"]) == (cell, arm)]
        available = [r for r in group if r["status"] == "complete"]
        n = len(available)
        rate = rejection_rate(
            sum(r["aic_k"] > 1 for r in available), n, len(group), family_size=family
        )
        result = {
            "cell": cell,
            "case": group[0]["case"],
            "n": group[0]["n"],
            "arm": arm,
            "primary_aic_multiple_components": rate,
            **{
                kind + "_k_counts": {
                    str(k): sum(r[kind + "_k"] == k for r in available) for k in range(1, 5)
                }
                for kind in ("aic", "bic")
            },
        }
        for kind in ("aic", "bic"):
            gains = []
            for r in available:
                k = r[kind + "_k"]
                metric = next(c for c in r["metrics"]["candidates"] if c["k"] == k)
                gains.append(metric["heldout_mean_logdensity_gain_over_one"])
            result[kind + "_selected_heldout_gain"] = describe(gains)
        result["negative_aic_selected_heldout_gain"] = sum(
            next(c for c in r["metrics"]["candidates"] if c["k"] == r["aic_k"])[
                "heldout_mean_logdensity_gain_over_one"
            ]
            < 0
            for r in available
        )
        cells.append(result)
    paired = []
    pairs = len(config.cases) * len(config.sizes) * 2
    for case in config.cases:
        for n in config.sizes:
            cell = f"{case}_n{n}"
            for arm in ARMS[1:]:

                def decisions(name, cell=cell):
                    return {
                        r["repetition"]: (r["aic_k"] > 1 if r["status"] == "complete" else None)
                        for r in rows
                        if r["cell"] == cell and r["arm"] == name
                    }

                record = paired_decisions(decisions(arm), decisions("baseline"), error=0.05 / pairs)
                record = {
                    k.replace("known", "baseline")
                    .replace("fitted", "control")
                    .replace("rejections", "events")
                    .replace("both_do_not_reject", "neither_event")
                    .replace("both_reject", "both_events"): v
                    for k, v in record.items()
                }
                paired.append(
                    {
                        "cell": cell,
                        "control_arm": arm,
                        "family_size": pairs,
                        "difference_scope": (
                            "Baseline minus control; exact marginal "
                            "discordance bounds adjusted over prescribed comparisons"
                        ),
                        **record,
                    }
                )
    return {
        "scope": (
            "Gaussian-control component selection and score-space prediction; "
            "no new p-values or species claims"
        ),
        "point_fits": len(rows),
        "successful_fits": sum(r["status"] == "complete" for r in rows),
        "failed_fits": sum(r["status"] != "complete" for r in rows),
        "primary_family_size": family,
        "cells": cells,
        "paired": paired,
        "no_audit_refits": True,
        "calibration_draws": 0,
        "stability_draws": 0,
        "interval_scope": (
            "Exact simultaneous intervals for independent-dataset "
            "K>1 rates; not species uncertainty"
        ),
        "prediction_scope": (
            "Independent test data through each frozen training "
            "map; gains within each score space only"
        ),
    }


def audit_study(source, *, output_dir, scratch, workers=1):
    root, destination = Path(source).resolve(), Path(output_dir).resolve()
    if root == destination or root.is_relative_to(destination) or destination.is_relative_to(root):
        raise ValueError("Audit output must be separate from immutable source")
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four one-thread audit workers")
    config = AICDiagnosticConfig.from_dict(read_json(root / "configuration.json"))
    plan = config.plan()
    if (
        read_json(root / "plan.json") != plan
        or read_json(root / "summary.json")["status"] != "complete"
    ):
        raise ValueError("Study plan or completion differs")
    ledger = [json.loads(s) for s in (root / "observations.jsonl").read_text().splitlines()]
    mapping = {(r["cell"], r["repetition"]): r for r in ledger}
    if len(mapping) != len(ledger) or set(mapping) != {(r["cell"], r["repetition"]) for r in plan}:
        raise ValueError("Missing, duplicate or unplanned ledger records")
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    tasks = [
        (
            str(root),
            r,
            mapping[r["cell"], r["repetition"]],
            config.to_dict(),
            str(Path(scratch).resolve()),
        )
        for r in plan
    ]
    with (
        ProcessPoolExecutor(max_workers=workers) as executor,
        gzip.open(destination / "points.jsonl.gz", "xt") as handle,
    ):
        for group in executor.map(audit_observation, tasks, chunksize=1):
            for row in group:
                rows.append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            write_progress(
                {
                    "point_fits_audited": len(rows),
                    "requested": sum(r["planned_fits"] for r in plan),
                },
                destination / "progress.json",
            )
    result = summarize(rows, config)
    execute = read_json(root / "summary.json")
    if (
        execute["successful_fits"] != result["successful_fits"]
        or execute["attempted_fits"] != result["point_fits"]
    ):
        raise ValueError("Independent accounting differs")
    write_json(result, destination / "summary.json")
    return result
