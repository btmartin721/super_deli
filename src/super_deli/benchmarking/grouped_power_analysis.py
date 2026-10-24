"""Prespecified power, recovery, overpartitioning and stability performance summaries."""

from __future__ import annotations

import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from super_deli.benchmarking.grouped_false_positive import paired_decisions, rejection_rate
from super_deli.benchmarking.grouped_power_audit import audit_observation
from super_deli.benchmarking.grouped_power_study import COUNT_KEYS, SCOPE
from super_deli.benchmarking.local_archives import write_progress
from super_deli.io.config import read_json, write_json
from super_deli.models.grouped_power_config import GroupedPowerStudyConfig


def describe(values):
    import numpy as np

    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite performance metric cannot be silently omitted")
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "q25": None, "q75": None, "mcse_mean": None}
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "mcse_mean": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
    }


def _rate(values, requested, *, family=1):
    values = [v for v in values if v is not None]
    if any(type(v) is not bool for v in values):
        raise ValueError("Binary performance endpoints must be Boolean or unavailable")
    result = rejection_rate(sum(values), len(values), requested, family_size=family)
    if family == 1:
        result.pop("bonferroni_familywise_95")
        result["interval_scope"] = "Pointwise descriptive interval; no familywise claim"
    return result


def _paired(rows, left, right, names):
    result = paired_decisions(
        {r["repetition"]: left(r) for r in rows}, {r["repetition"]: right(r) for r in rows}
    )
    return {k.replace("known", names[1]).replace("fitted", names[0]): v for k, v in result.items()}


def summarize(rows, config):
    expected = {(r["cell"], r["repetition"]) for r in config.plan()}
    identities = [(r["cell"], r["repetition"]) for r in rows]
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError("Missing, repeated or unplanned observation identities")
    primary, cells = [], []
    family = 3 * sum(c.has_groups for c in config.cells)
    for cell in config.cells:
        group = sorted((r for r in rows if r["cell"] == cell.name), key=lambda r: r["repetition"])
        n = cell.n_datasets
        metric = {
            "cell": cell.name,
            "law": cell.law,
            "signal": cell.signal,
            "n_units": cell.n_units,
            "requested": n,
            "has_groups": cell.has_groups,
            "interpretation": "Constructed four-group alternative"
            if cell.has_groups
            else "Declared Gaussian null screening control"
            if cell.law == "independent_gaussian"
            else ("No discrete groups; Gaussian-departure stress, not species/type-I error"),
            "likelihood_rejection": _rate([r["likelihood_reject"] for r in group], n),
            "bic_rejection": _rate([r["bic_reject"] for r in group], n),
        }
        for criterion in ("aic", "bic"):
            available = [r for r in group if r[criterion + "_k"] is not None]
            metric[criterion + "_selected_k_counts"] = {
                str(k): sum(r[criterion + "_k"] == k for r in available)
                for k in range(1, config.calibration.model.max_k + 1)
            }
            metric[criterion + "_multiple_components"] = _rate(
                [
                    r[criterion + "_k"] > 1 if r[criterion + "_k"] is not None else None
                    for r in group
                ],
                n,
            )
            metric[criterion + "_ari"] = describe([r[criterion + "_ari"] for r in group])
        metric["paired_likelihood_vs_bic_rejection"] = _paired(
            group,
            lambda r: r["bic_reject"],
            lambda r: r["likelihood_reject"],
            ("bic", "likelihood"),
        )
        if cell.has_groups:
            events = {
                "likelihood_rejection": [r["likelihood_reject"] for r in group],
                "aic_k4": [r["aic_k"] == 4 if r["aic_k"] is not None else None for r in group],
                "joint_rejection_and_aic_k4": [
                    (r["likelihood_reject"] and r["aic_k"] == 4)
                    if r["likelihood_reject"] is not None and r["aic_k"] is not None
                    else None
                    for r in group
                ],
            }
            for endpoint, outcomes in events.items():
                primary.append(
                    {"cell": cell.name, "endpoint": endpoint, **_rate(outcomes, n, family=family)}
                )
            metric["bic_k4"] = _rate(
                [r["bic_k"] == 4 if r["bic_k"] is not None else None for r in group], n
            )
            metric["paired_aic_vs_bic_k4"] = _paired(
                group,
                lambda r: r["bic_k"] == 4 if r["bic_k"] is not None else None,
                lambda r: r["aic_k"] == 4 if r["aic_k"] is not None else None,
                ("bic", "aic"),
            )
            metric["aic_ari_given_rejection_secondary"] = describe(
                [r["aic_ari"] for r in group if r["likelihood_reject"] is True]
            )
            stability = [r["stability"] for r in group if r["stability"] is not None]
            metric["stability"] = {
                "requested_observations": config.stability_first,
                "verified_point_replays": sum(s["point_replay_verified"] for s in stability),
                "requested_draws": sum(s["requested"] for s in stability),
                "successful_draws": sum(s["successful"] for s in stability),
                "failed_draws": sum(s["failed"] for s in stability),
                "coassignment_truth_mse_across_observations": describe(
                    [s.get("coassignment_truth_mse") for s in stability]
                ),
                "bootstrap_ari_mean_across_observations": describe(
                    [s.get("bootstrap_ari_mean") for s in stability]
                ),
                "scope": (
                    "First prespecified observations; conditional repeatability, not species "
                    "confidence. Locality pairs are not independent study replicates."
                ),
            }
        cells.append(metric)
    return {
        "scope": SCOPE,
        "observations": len(rows),
        "cells": cells,
        "primary_rates": primary,
        "primary_family_size": family,
        "multiple_comparisons": (
            "Exact Bonferroni intervals across three endpoints by six alternative cells "
            "in the approved design; all other summaries are descriptive"
        ),
        "fit_accounting": {key: sum(r["accounting"][key] for r in rows) for key in COUNT_KEYS},
        "numerically_audited_observations": sum(
            r["audit_status"] == "complete_numerical_audit" for r in rows
        ),
        "archive_members": sum(r["archive_members"] for r in rows),
        "truth_scope": (
            "Post-fit generating partitions; no species-boundary or selected-K significance claim"
        ),
        "stability_audit_scope": (
            "Seeds, whole-group membership/counts, PCA moments, candidate criteria, "
            "partitions, coassignment and all replay-point densities. Production exports"
            " no per-resample Gaussian parameter arrays."
        ),
        "no_model_refits": True,
    }


def audit_study(root, *, output_dir, scratch, workers=1):
    root, destination = Path(root).resolve(), Path(output_dir).resolve()
    if destination == root or destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError("Keep audit output separate from the immutable compact study")
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four independent one-thread audit workers")
    config = GroupedPowerStudyConfig.from_dict(read_json(root / "study.json"))
    plan = config.plan()
    if (
        read_json(root / "plan.json") != plan
        or read_json(root / "summary.json")["status"] != "complete"
    ):
        raise ValueError("Study declaration/plan/completion differs")
    ledger = [json.loads(s) for s in (root / "observations.jsonl").read_text().splitlines()]
    mapping = {(r["cell"], r["repetition"]): r for r in ledger}
    if len(mapping) != len(ledger) or set(mapping) != {(r["cell"], r["repetition"]) for r in plan}:
        raise ValueError("Execution ledger contains missing/replaced observations")
    destination.mkdir(parents=True, exist_ok=False)
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
    rows = []
    with (
        ProcessPoolExecutor(max_workers=workers) as executor,
        gzip.open(destination / "pipelines.jsonl.gz", "xt", encoding="utf-8") as points,
        (destination / "observations.jsonl").open("x", encoding="utf-8") as observations,
    ):
        for result in executor.map(audit_observation, tasks, chunksize=1):
            pipelines = result.pop("pipelines")
            for pipeline in pipelines:
                points.write(json.dumps(pipeline, sort_keys=True) + "\n")
            observations.write(json.dumps(result, sort_keys=True) + "\n")
            observations.flush()
            rows.append(result)
            write_progress(
                {"observations_audited": len(rows), "requested": len(tasks)},
                destination / "progress.json",
            )
    result = summarize(rows, config)
    if result["fit_accounting"] != {key: sum(r[key] for r in ledger) for key in COUNT_KEYS}:
        raise ValueError("Aggregate fit accounting differs")
    expected = read_json(root / "summary.json")
    if any(expected[key] != result["fit_accounting"][key] for key in COUNT_KEYS):
        raise ValueError("Execution summary fit accounting differs")
    result["retained_archive_bytes"] = sum(r["size_bytes"] for r in ledger)
    write_json(result, destination / "summary.json")
    return result
