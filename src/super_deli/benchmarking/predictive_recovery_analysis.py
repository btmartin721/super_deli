"""Independent fresh-observation audits and dataset-level paired summaries."""

import gzip
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from super_deli.benchmarking.grouped_false_positive import paired_decisions, rejection_rate
from super_deli.benchmarking.grouped_power_analysis import describe
from super_deli.benchmarking.local_archives import materialize, write_progress
from super_deli.benchmarking.predictive_audit import independent_simulation
from super_deli.benchmarking.predictive_recovery_audit import audit_partition, close
from super_deli.io.config import read_json, write_json
from super_deli.models.predictive_recovery_config import PredictiveRecoveryConfig
from super_deli.models.predictive_study_config import PredictiveCell

COUNTS = (
    "planned_searches",
    "attempted_searches",
    "successful_searches",
    "failed_searches",
    "unattempted_searches",
    "planned_oracles",
    "attempted_oracles",
    "successful_oracles",
    "failed_oracles",
    "unattempted_oracles",
)


def audit_oracles(base, truth, tests, test_layout, model):
    import numpy as np
    from scipy.special import logsumexp
    from scipy.stats import multivariate_normal

    from super_deli.benchmarking.full_grouped_oracle import audit_oracle
    from super_deli.benchmarking.predictive_audit import project_independent
    from super_deli.workflows.predictive import fold_model

    workflow = base / "part_000/workflow"
    results = []
    assignments = (
        read_json(workflow / "fold_assignments.json")
        if (workflow / "fold_assignments.json").exists()
        else []
    )
    definitions = [("full", None, model)] + [
        (f"fold_{a['fold']:03d}", a["training_indices"], fold_model(model, a["fold"]))
        for a in assignments
    ]
    for prefix, indices, trained in definitions:
        directory = base / "oracles" / prefix
        if not (directory / "result.json").exists():
            continue
        record = read_json(directory / "result.json")
        run = workflow / prefix / "run"
        with np.load(run / "gmm_candidates.npz", allow_pickle=False) as pool:
            scores = pool["scores"]
            labels = truth if indices is None else truth[indices]
            if set(labels.tolist()) == {0, 1, 2, 3}:
                audit_oracle(directory, labels, scores, "four", trained, record)
            elif record["status"] != "failed" or not record.get("error"):
                raise ValueError("Unsupported oracle initialization not retained")
        if record["status"] == "complete":
            with (
                np.load(directory / "mixture.npz", allow_pickle=False) as p,
                np.load(directory / "predictions.npz", allow_pickle=False) as predictions,
            ):

                def logpdf(x):
                    return logsumexp(
                        np.column_stack(
                            [
                                np.log(w) + multivariate_normal.logpdf(x, mean=m, cov=c)
                                for w, m, c in zip(
                                    p["weights"], p["means"], p["covariances"], strict=True
                                )
                            ]
                        ),
                        axis=1,
                    )

                external = project_independent(run, tests, test_layout)
                np.testing.assert_allclose(predictions["external_scores"], external, atol=1e-10)
                np.testing.assert_allclose(
                    predictions["external_logdensity"], logpdf(external), atol=1e-8
                )
                np.testing.assert_allclose(
                    predictions["training_logdensity"], logpdf(scores), atol=1e-8
                )
                train_ll = float(logpdf(scores).sum())
                one = next(c for c in read_json(run / "candidates.json") if c["k"] == 1)
                four = next(c for c in read_json(run / "candidates.json") if c["k"] == 4)
                with np.load(
                    base / "part_000/diagnostics" / (prefix + "_external.npz"), allow_pickle=False
                ) as held:
                    external_gain = float((logpdf(external) - held["k1"]).mean())
                if indices is not None:
                    with np.load(workflow / prefix / "validation.npz", allow_pickle=False) as held:
                        np.testing.assert_allclose(
                            predictions["validation_scores"], held["scores"], atol=1e-11
                        )
                        np.testing.assert_allclose(
                            predictions["validation_logdensity"], logpdf(held["scores"]), atol=1e-8
                        )
                record = {
                    **record,
                    "loglikelihood_gain_over_unsupervised_k4": train_ll - four["log_likelihood"],
                    "aic": -2 * train_ll + 2 * (4 * (4 + 4 * 5 // 2) + 3),
                    "k1_aic": one["aic"],
                    "external_mean_logdensity_gain": external_gain,
                }
        results.append({"prefix": prefix, **record})
    return results


def _audit(task):
    import numpy as np

    source, declaration, row, raw_config, scratch = task
    config = PredictiveRecoveryConfig.from_dict(raw_config)
    study = config.study
    for key, value in declaration.items():
        close(row[key], value)
    cell = PredictiveCell.from_dict(declaration["cell_config"])
    with materialize(
        Path(source) / row["archive"], row, scratch=scratch, min_free_bytes=study.min_free_bytes
    ) as base:
        close(read_json(base / "declaration.json"), declaration)
        original = read_json(base / "record.json")
        for key, value in original.items():
            close(row[key], value)
        values, layout, truth = independent_simulation(cell, cell.n_units, declaration["data_seed"])
        tests, test_layout, test_truth = independent_simulation(
            cell, study.n_test, declaration["test_seed"]
        )
        layout = json.loads(json.dumps(layout))
        test_layout = json.loads(json.dumps(test_layout))
        test_layout["unit_ids"] = ["heldout_" + u for u in test_layout["unit_ids"]]
        for specimen in test_layout["specimens"]:
            for k in specimen:
                specimen[k] = "heldout_" + specimen[k]
        for view in test_layout["views"]:
            for key in ("observation_ids", "unit_ids", "target_ids"):
                view[key] = ["heldout_" + u for u in view[key]]
        close(read_json(base / "layout.json"), layout)
        close(read_json(base / "external_layout.json"), test_layout)
        for name, expected in (("values.npz", values), ("external_values.npz", tests)):
            with np.load(base / name, allow_pickle=False) as p:
                for k in expected:
                    np.testing.assert_allclose(p[k], expected[k], atol=4e-14, rtol=4e-14)
                    expected[k] = p[k].copy()
        with np.load(base / "truth.npz", allow_pickle=False) as p:
            np.testing.assert_array_equal(p["training"], truth)
            np.testing.assert_array_equal(p["external"], test_truth)
        names = [f"part_{i:03d}" for i in range(config.partitions)] + ["part_extended"]
        close([p["name"] for p in row["parts"]], names)
        unique = []
        for i, part in enumerate(row["parts"]):
            name = part["name"]
            directory = base / name
            extended = name == "part_extended"
            model = replace(study.model, n_init=config.extended_starts) if extended else study.model
            seed = declaration["partition_seeds"][0 if extended else i]
            if part["screen_seed"] != seed or part["n_init"] != model.n_init:
                raise ValueError("Arm settings differ")
            reuse = not extended and i > 0
            if part["shared_full_search"] != reuse:
                raise ValueError("Shared-fit declaration differs")
            if reuse and (directory / "workflow/full").exists():
                a = base / "part_000/workflow/full"
                b = directory / "workflow/full"
                left = {
                    p.relative_to(a).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in a.rglob("*")
                    if p.is_file()
                }
                right = {
                    p.relative_to(b).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in b.rglob("*")
                    if p.is_file()
                }
                if left != right:
                    raise ValueError("Reused full fit differs from shared original")
            if not (directory / "diagnostic_failure.json").exists():
                summary = audit_partition(
                    directory,
                    values,
                    layout,
                    truth if cell.role == "alternative" else None,
                    tests,
                    test_layout,
                    model,
                    seed,
                    5,
                )
                close(part["selection"], summary["selection"])
                close(part["performance"], read_json(directory / "performance.json"))
                close(part["gates"], read_json(directory / "diagnostics/metrics.json")["gates"])
            elif part["selection"]["status"] != "withheld" or not read_json(
                directory / "diagnostic_failure.json"
            ).get("error"):
                raise ValueError("Diagnostic failure not withheld with reason")
            for p in (directory / "workflow").glob("*/record.json"):
                if p.parent.name != "full" or not reuse:
                    unique.append(read_json(p))
        oracles = (
            audit_oracles(base, truth, tests, test_layout, study.model)
            if cell.role == "alternative"
            else []
        )
        counts = {
            "planned_searches": declaration["planned_searches"],
            "attempted_searches": len(unique),
            "successful_searches": sum(r["status"] == "complete" for r in unique),
            "failed_searches": sum(r["status"] != "complete" for r in unique),
            "unattempted_searches": declaration["planned_searches"] - len(unique),
            "planned_oracles": declaration["planned_oracles"],
            "attempted_oracles": len(oracles),
            "successful_oracles": sum(r["status"] == "complete" for r in oracles),
            "failed_oracles": sum(r["status"] != "complete" for r in oracles),
            "unattempted_oracles": declaration["planned_oracles"] - len(oracles),
        }
        for key, value in counts.items():
            close(row[key], value)
        return {**row, "oracles": oracles}


def audit_observation(task):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _audit(task)


def summarize(rows, config):
    if {(r["cell"], r["repetition"]) for r in rows} != {
        (r["cell"], r["repetition"]) for r in config.plan()
    } or len(rows) != len(config.plan()):
        raise ValueError("Missing/duplicate/unplanned observations")
    cells = []
    rates = []
    paired = []
    family = 4 * len(config.study.cells)
    for cell in config.study.cells:
        group = [r for r in rows if r["cell"] == cell.name]
        methods = {}
        decisions = {}
        for arm in ("baseline", "extended"):
            for method in ("ordinary", "screened"):
                name = arm + "_" + method
                partname = "part_000" if arm == "baseline" else "part_extended"
                values = [
                    next(p for p in r["parts"] if p["name"] == partname)["performance"][method]
                    for r in group
                ]
                good = [v for v in values if v is not None]
                events = sum(
                    v["k"] == 4 if cell.role == "alternative" else v["k"] > 1 for v in good
                )
                rate = rejection_rate(events, len(good), cell.n_datasets, family_size=family)
                methods[name] = {
                    "rate": rate,
                    "k_counts": {str(k): sum(v["k"] == k for v in good) for k in range(1, 5)},
                    "ari": describe([v["ari"] for v in good]),
                    "external_gain": describe([v["external_mean_logdensity_gain"] for v in good]),
                }
                rates.append(
                    {
                        "cell": cell.name,
                        "method": name,
                        "event": "K=4" if cell.role == "alternative" else "K>1",
                        **rate,
                    }
                )
                decisions[name] = {
                    r["repetition"]: None
                    if v is None
                    else (v["k"] == 4 if cell.role == "alternative" else v["k"] > 1)
                    for r, v in zip(group, values, strict=True)
                }
        for a, b in (
            ("baseline_ordinary", "baseline_screened"),
            ("baseline_screened", "extended_screened"),
        ):
            p = paired_decisions(
                decisions[b], decisions[a], error=0.05 / (2 * len(config.study.cells))
            )
            renamed = {
                k.replace("known", "reference")
                .replace("fitted", "comparison")
                .replace("rejections", "events"): v
                for k, v in p.items()
            }
            paired.append(
                {
                    "cell": cell.name,
                    "reference": a,
                    "comparison": b,
                    "family_size": 2 * len(config.study.cells),
                    **renamed,
                }
            )
        repeats = []
        for row in group:
            parts = [p for p in row["parts"] if p["name"] != "part_extended"]
            good = [
                p["selection"]["screened_k"]
                for p in parts
                if p["selection"]["status"] == "complete"
            ]
            repeats.append(
                {
                    "repetition": row["repetition"],
                    "requested": config.partitions,
                    "available": len(good),
                    "k4_selections": sum(k == 4 for k in good),
                    "k_gt1": sum(k > 1 for k in good),
                    "distinct_selections": len(set(good)),
                }
            )
        cells.append({"cell": cell.to_dict(), "methods": methods, "split_sensitivity": repeats})
    return {
        "observations": len(rows),
        **{k: sum(r[k] for r in rows) for k in COUNTS},
        "primary_family_size": family,
        "primary_rates": rates,
        "paired": paired,
        "cells": cells,
        "withheld_parts": sum(
            p["selection"]["status"] != "complete" for r in rows for p in r["parts"]
        ),
        "scope": (
            "Diagnosis only; repeated partitions are not independent datasets "
            "or biological stability"
        ),
        "calibration_draws": 0,
        "stability_draws": 0,
        "no_audit_refits": True,
    }


def audit_study(source, *, output_dir, scratch, workers=1):
    root = Path(source).resolve()
    destination = Path(output_dir).resolve()
    if root == destination or root.is_relative_to(destination) or destination.is_relative_to(root):
        raise ValueError("Keep immutable evidence and audit separate")
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four workers")
    c = PredictiveRecoveryConfig.from_dict(read_json(root / "configuration.json"))
    plan = c.plan()
    if (
        read_json(root / "plan.json") != plan
        or read_json(root / "summary.json")["status"] != "complete"
    ):
        raise ValueError("Study plan/completion differs")
    ledger = [json.loads(s) for s in (root / "observations.jsonl").read_text().splitlines()]
    mapping = {(r["cell"], r["repetition"]): r for r in ledger}
    if len(mapping) != len(ledger) or set(mapping) != {(r["cell"], r["repetition"]) for r in plan}:
        raise ValueError("Ledger identities differ")
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    with (
        ProcessPoolExecutor(max_workers=workers) as pool,
        gzip.open(destination / "points.jsonl.gz", "xt") as out,
    ):
        tasks = (
            (
                str(root),
                r,
                mapping[r["cell"], r["repetition"]],
                c.to_dict(),
                str(Path(scratch).resolve()),
            )
            for r in plan
        )
        for row in pool.map(audit_observation, tasks, chunksize=1):
            rows.append(row)
            out.write(json.dumps(row, sort_keys=True) + "\n")
            write_progress(
                {"observations_audited": len(rows), "requested": len(plan)},
                destination / "progress.json",
            )
    result = summarize(rows, c)
    for key in COUNTS:
        close(result[key], read_json(root / "summary.json")[key])
    write_json(result, destination / "summary.json")
    return result
