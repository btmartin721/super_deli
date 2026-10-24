"""Independent paired-family audits and dataset-level uncertainty summaries."""

from __future__ import annotations

import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from super_deli.benchmarking.grouped_false_positive import paired_decisions, rejection_rate
from super_deli.benchmarking.grouped_power_analysis import describe
from super_deli.benchmarking.local_archives import materialize, write_progress
from super_deli.benchmarking.predictive_audit import audit_observation as audit_arm
from super_deli.benchmarking.shared_predictive_study import COUNTS, same_maps
from super_deli.io.config import read_json, write_json
from super_deli.models.shared_predictive_config import SharedPredictiveConfig


def audit_external_folds(base, model):
    import numpy as np

    from super_deli.benchmarking.aic_diagnostic_audit import check_pool
    from super_deli.benchmarking.predictive_audit import project_independent

    with np.load(base / "external_values.npz", allow_pickle=False) as p:
        values = {k: p[k] for k in p.files}
    layout = read_json(base / "external_layout.json")
    for row in read_json(base / "external_folds.json"):
        run = base / "workflow" / row["prefix"] / "run"
        if row["status"] != "complete":
            if row["status"] == "unavailable" and (run / "gmm_candidates.npz").exists():
                raise ValueError("Available external fold marked unavailable")
            if row["status"] == "failed" and not row.get("error"):
                raise ValueError("External failure lacks reason")
            continue
        expected = project_independent(run, values, layout)
        with np.load(base / (row["prefix"] + "_external.npz"), allow_pickle=False) as held:
            np.testing.assert_allclose(held["scores"], expected, atol=1e-11, rtol=1e-11)
            _, density = check_pool(
                (run / "gmm_candidates.npz").read_bytes(),
                read_json(run / "candidates.json"),
                counts=None,
                config=model,
                heldout=held,
            )
            for k, v in row["gains"].items():
                np.testing.assert_allclose(v, (density[k] - density["1"]).mean(), atol=1e-10)


def _exported(base, family):
    import numpy as np
    from scipy.special import logsumexp
    from scipy.stats import multivariate_normal

    run = base / "workflow"
    summary = read_json(run / "summary.json")
    k = summary["selection"].get("screened_k")
    if k is None:
        if (run / "selected_model.npz").exists():
            raise ValueError("Unavailable selection exported a model")
        return
    with (
        np.load(run / "selected_model.npz", allow_pickle=False) as p,
        np.load(run / "full/run/gmm_candidates.npz", allow_pickle=False) as pool,
        np.load(run / "selected_memberships.npz", allow_pickle=False) as saved,
    ):
        if family == "tied":
            if str(p["covariance_type"]) != family:
                raise ValueError("Selected family declaration differs")
            np.testing.assert_array_equal(
                p["covariances"], np.repeat(p["covariances"][:1], k, axis=0)
            )
        terms = np.column_stack(
            [
                np.log(a) + multivariate_normal.logpdf(pool["scores"], mean=m, cov=v)
                for a, m, v in zip(p["weights"], p["means"], p["covariances"], strict=True)
            ]
        )
        probabilities = np.exp(terms - logsumexp(terms, axis=1)[:, None])
        np.testing.assert_allclose(saved["membership"], probabilities, atol=1e-10)
        np.testing.assert_array_equal(saved["labels"], probabilities.argmax(1))


def audit_observation(task):
    from threadpoolctl import threadpool_limits

    source, raw, declaration, row, scratch = task
    c = SharedPredictiveConfig.from_dict(raw)
    if any(row[k] != v for k, v in declaration.items()):
        raise ValueError("Paired ledger declaration differs")
    with (
        threadpool_limits(limits=1),
        materialize(
            Path(source) / row["archive"],
            row,
            scratch=scratch,
            min_free_bytes=c.study.min_free_bytes,
        ) as base,
    ):
        if read_json(base / "declaration.json") != declaration:
            raise ValueError("Archived declaration differs")
        original = read_json(base / "record.json")
        if any(row[k] != v for k, v in original.items()):
            raise ValueError("Archived record differs")
        records = {}
        for family in ("full", "tied"):
            part = {**declaration, "planned_fits": declaration["planned_fits"] // 2}
            r = row["arms"][family]
            records[family] = audit_arm(
                (str(base / family), part, r, c.family(family).to_dict(), scratch)
            )
            with materialize(base / family / r["archive"], r, scratch=scratch) as arm:
                _exported(arm, family)
                audit_external_folds(arm, c.family(family).model)
        left, right = row["arms"]["full"], row["arms"]["tied"]
        with (
            materialize(base / "full" / left["archive"], left, scratch=scratch) as a,
            materialize(base / "tied" / right["archive"], right, scratch=scratch) as b,
        ):
            compared = same_maps(a, b)
            if row["map_comparisons"] != compared or row["maps_verified_identical"] != (
                compared == 6
            ):
                raise ValueError("Paired map availability differs")
        for key in COUNTS:
            if row[key] != sum(v[key] for v in records.values()):
                raise ValueError("Paired search accounting differs")
    return row


def summarize(rows, config):
    expected = {(r["cell"], r["repetition"]) for r in config.plan()}
    if len(rows) != len(expected) or {(r["cell"], r["repetition"]) for r in rows} != expected:
        raise ValueError("Missing/duplicate/unplanned paired observations")
    nprimary = sum(c.role != "stress" for c in config.study.cells)
    rates, pairs, cells = [], [], []
    for cell in config.study.cells:
        group = [r for r in rows if r["cell"] == cell.name]
        methods, decisions = {}, {}
        for family in ("full", "tied"):
            for method in ("ordinary", "screened"):
                name = family + "_" + method
                raw = [r["arms"][family]["performance"][method] for r in group]
                good = [v for v in raw if v is not None]

                def event(v, role=cell.role):
                    return v["k"] == 4 if role == "alternative" else v["k"] > 1

                rate = rejection_rate(
                    sum(event(v) for v in good),
                    len(good),
                    cell.n_datasets,
                    family_size=max(1, 4 * nprimary),
                )
                methods[name] = {
                    "rate": rate,
                    "k_counts": {str(k): sum(v["k"] == k for v in good) for k in range(1, 5)},
                    "ari": describe([v["ari"] for v in good]),
                    "external_gain": describe([v["external_mean_logdensity_gain"] for v in good]),
                }
                decisions[name] = {
                    r["repetition"]: None if v is None else bool(event(v))
                    for r, v in zip(group, raw, strict=True)
                }
                if cell.role != "stress":
                    rates.append(
                        {
                            "cell": cell.name,
                            "method": name,
                            "event": "K=4" if cell.role == "alternative" else "K>1",
                            **rate,
                        }
                    )
        if cell.role != "stress":
            for reference, comparison in (
                ("full_ordinary", "full_screened"),
                ("tied_ordinary", "tied_screened"),
                ("full_screened", "tied_screened"),
            ):
                paired = paired_decisions(
                    decisions[comparison], decisions[reference], error=0.05 / (3 * nprimary)
                )
                paired = {
                    k.replace("known", "reference")
                    .replace("fitted", "comparison")
                    .replace("rejections", "events"): v
                    for k, v in paired.items()
                }
                pairs.append(
                    {
                        "cell": cell.name,
                        "reference": reference,
                        "comparison": comparison,
                        "family_size": 3 * nprimary,
                        **paired,
                    }
                )
        cells.append({"cell": cell.to_dict(), "methods": methods})
    return {
        "observations": len(rows),
        **{k: sum(r[k] for r in rows) for k in COUNTS},
        "screening_withheld": sum(r["screening_withheld"] for r in rows),
        "external_fold_failures": sum(r["external_fold_failures"] for r in rows),
        "primary_family_size": 4 * nprimary,
        "primary_rates": rates,
        "paired": pairs,
        "cells": cells,
        "calibration_draws": 0,
        "stability_draws": 0,
        "no_audit_refits": True,
        "scope": "Fixed-family predictive heuristic; no family selection or species significance",
    }


def audit_study(source, *, output_dir, scratch, workers=1):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four workers")
    source, output = Path(source).resolve(), Path(output_dir).resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Keep immutable evidence and audit separate")
    c = SharedPredictiveConfig.from_dict(read_json(source / "configuration.json"))
    if (
        read_json(source / "plan.json") != c.plan()
        or read_json(source / "summary.json")["status"] != "complete"
    ):
        raise ValueError("Incomplete/unplanned paired study")
    rows = [json.loads(s) for s in (source / "observations.jsonl").read_text().splitlines()]
    mapping = {(r["cell"], r["repetition"]): r for r in rows}
    if len(mapping) != len(rows) or set(mapping) != {
        (r["cell"], r["repetition"]) for r in c.plan()
    }:
        raise ValueError("Ledger identities differ")
    output.mkdir(parents=True, exist_ok=False)
    audited = []
    with (
        ProcessPoolExecutor(max_workers=workers) as pool,
        gzip.open(output / "points.jsonl.gz", "xt") as handle,
    ):
        tasks = (
            (
                str(source),
                c.to_dict(),
                d,
                mapping[d["cell"], d["repetition"]],
                str(Path(scratch).resolve()),
            )
            for d in c.plan()
        )
        for row in pool.map(audit_observation, tasks, chunksize=1):
            audited.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            write_progress(
                {"observations_audited": len(audited), "requested": len(rows)},
                output / "progress.json",
            )
    result = summarize(audited, c)
    original = read_json(source / "summary.json")
    if any(
        result[k] != original[k] for k in (*COUNTS, "screening_withheld", "external_fold_failures")
    ):
        raise ValueError("Independent aggregate accounting differs")
    write_json(result, output / "summary.json")
    return result
