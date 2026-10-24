"""Read-only archived candidate audits and explicitly counted point replays."""

from __future__ import annotations

import gzip
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

from super_deli.benchmarking.aic_diagnostic_audit import check_metrics
from super_deli.benchmarking.aic_diagnostic_metrics import candidate_metrics
from super_deli.benchmarking.grouped_likelihood_analysis import audit_candidates
from super_deli.benchmarking.grouped_power_study import _verify_replay
from super_deli.benchmarking.grouped_study import simulate_grouped
from super_deli.benchmarking.local_archives import compact_directory, materialize, write_progress
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json
from super_deli.models.aic_diagnostic_config import AICDiagnosticConfig
from super_deli.models.integrative_config import IntegrativeConfig

OLD_CASES = (
    "independent_gaussian",
    "correlated_gaussian",
    "zero_conditional_gaussian",
    "rank_two_gaussian",
    "weak_conditional_gaussian",
)


def declarations(null_study, power_study):
    tasks = []
    null_study, power_study = Path(null_study).resolve(), Path(power_study).resolve()
    for case in OLD_CASES:
        root = null_study / case
        configuration = read_json(root / "study.json")
        model = IntegrativeConfig.from_dict(configuration["calibration"]["model"])
        ledger = [json.loads(s) for s in (root / "datasets.jsonl").read_text().splitlines()]
        if len(ledger) != configuration["n_datasets"] or sorted(
            r["repetition"] for r in ledger
        ) != list(range(len(ledger))):
            raise ValueError("Archived cohort identities differ")
        for row in ledger:
            tasks.append(
                {
                    "cohort": "larger_null_validation",
                    "case": case,
                    "n": 64,
                    "repetition": row["repetition"],
                    "data_seed": row["data_seed"],
                    "source": str(root / f"dataset_{row['repetition']:04d}"),
                    "model": model.to_dict(),
                }
            )
    configuration = read_json(power_study / "study.json")
    model = IntegrativeConfig.from_dict(configuration["calibration"]["model"])
    for row in map(json.loads, (power_study / "observations.jsonl").read_text().splitlines()):
        if row["cell"] in {"null_n64", "null_n256"}:
            tasks.append(
                {
                    "cohort": "power_controls",
                    "case": "independent_gaussian",
                    "n": row["cell_config"]["n_units"],
                    "repetition": row["repetition"],
                    "data_seed": row["data_seed"],
                    "source": str(power_study),
                    "ledger": row,
                    "model": model.to_dict(),
                }
            )
    return tasks


@contextmanager
def observation(task, config, scratch):
    if task["cohort"] == "power_controls":
        with materialize(
            Path(task["source"]) / task["ledger"]["archive"],
            task["ledger"],
            scratch=scratch,
            min_free_bytes=config.min_free_bytes,
        ) as base:
            yield base
    else:
        yield Path(task["source"])


def _source_task(arguments):
    import numpy as np
    from threadpoolctl import threadpool_limits

    task, raw_config, scratch = arguments
    config = AICDiagnosticConfig.from_dict(raw_config)
    model = IntegrativeConfig.from_dict(task["model"])
    if model.to_dict() != config.model.to_dict():
        raise ValueError("Archived point model differs from the frozen diagnostic baseline")
    with threadpool_limits(limits=1), observation(task, config, scratch) as base:
        values, layout, _ = simulate_grouped(task["case"], task["n"], task["data_seed"])
        with np.load(base / "values.npz", allow_pickle=False) as p:
            for name in values:
                np.testing.assert_array_equal(p[name], values[name])
        directory = base / "calibration"
        point = read_json(directory / "observed.json")
        source_inference = read_json(directory / "summary.json")["inference"]
        if point["status"] != "complete":
            return {
                "cohort": task["cohort"],
                "case": task["case"],
                "n": task["n"],
                "repetition": task["repetition"],
                "status": "retained_failure",
                "metrics": None,
            }
        with ZipFile(directory / "pipeline_artifacts.zip") as z:
            audit_candidates(
                z, point, SimpleNamespace(model=model), values, json.loads(json.dumps(layout))
            )
            raw = z.read("observed/run/gmm_candidates.npz")
            metrics, _ = candidate_metrics(
                raw,
                point["candidates"],
                counts=np.asarray(layout["counts"]),
                reg_covar=model.reg_covar,
            )
        check_metrics(raw, metrics, np.asarray(layout["counts"]), model.reg_covar)
        return {
            "cohort": task["cohort"],
            "case": task["case"],
            "n": task["n"],
            "repetition": task["repetition"],
            "status": "complete",
            "metrics": metrics,
            "old_p_value_unchanged": source_inference["p_value"],
            "old_reject_unchanged": source_inference.get("reject_grouped_raw_gaussian"),
            "observed_record_sha256": fingerprint_file(directory / "observed.json").sha256,
            "scope": "Existing inference copied read-only, never recalculated",
        }


def audit_sources(config, *, null_study, power_study, output_dir, scratch, workers=1):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four one-thread workers")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    tasks = declarations(null_study, power_study)
    write_json(tasks, root / "private-source-plan.json")
    rows = []
    with (
        ProcessPoolExecutor(max_workers=workers) as executor,
        gzip.open(root / "points.jsonl.gz", "xt") as handle,
    ):
        for row in executor.map(
            _source_task,
            ((r, config.to_dict(), str(Path(scratch).resolve())) for r in tasks),
            chunksize=1,
        ):
            rows.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            write_progress({"audited": len(rows), "requested": len(tasks)}, root / "progress.json")
    result = {
        "requested": len(tasks),
        "audited": sum(r["status"] == "complete" for r in rows),
        "groups": [],
        "new_significance_fits": 0,
        "source_models_unchanged": True,
    }
    for cohort, case, n in sorted({(r["cohort"], r["case"], r["n"]) for r in rows}):
        group = [r for r in rows if (r["cohort"], r["case"], r["n"]) == (cohort, case, n)]
        good = [r for r in group if r["status"] == "complete"]
        result["groups"].append(
            {
                "cohort": cohort,
                "case": case,
                "n": n,
                "requested": len(group),
                "available": len(good),
                **{
                    kind + "_k_counts": {
                        str(k): sum(r["metrics"][kind + "_k"] == k for r in good)
                        for k in range(1, 5)
                    }
                    for kind in ("aic", "bic")
                },
                "old_likelihood_rejected": sum(r["old_reject_unchanged"] is True for r in good),
                "aic_multiple_without_old_rejection": sum(
                    r["metrics"]["aic_k"] > 1 and r["old_reject_unchanged"] is False for r in good
                ),
            }
        )
    write_json(result, root / "summary.json")
    return result


def _replay(arguments):
    from threadpoolctl import threadpool_limits

    from super_deli.models.grouped_null import write_raw_views
    from super_deli.workflows.grouped_likelihood import point_fitter
    from super_deli.workflows.integrative import run_integrative

    root, task, raw_config, scratch = arguments
    config = AICDiagnosticConfig.from_dict(raw_config)
    if IntegrativeConfig.from_dict(task["model"]) != config.model:
        raise ValueError("Replay model differs from original")
    if shutil.disk_usage(root).free < config.min_free_bytes + 128 * 1024**2:
        raise RuntimeError("Insufficient replay working space")
    base = Path(root) / ".working" / f"n{task['n']}_{task['repetition']:04d}"
    base.mkdir(parents=True, exist_ok=False)
    values, layout, _ = simulate_grouped(task["case"], task["n"], task["data_seed"])
    dataset = write_raw_views(values, layout, base / "raw")
    verification_error = None
    with threadpool_limits(limits=1):
        record = point_fitter(
            base, dataset, base / "raw", "replay", model=config.model, runner=run_integrative
        )
        write_json(record, base / "record.json")
        if record["status"] == "complete":
            with observation(task, config, scratch) as source:
                try:
                    _verify_replay(source / "calibration", base / "run", config.model)
                except (ValueError, AssertionError, KeyError) as exc:
                    verification_error = str(exc)
                    write_json({"error": verification_error}, base / "verification_failure.json")
    relative = f"n{task['n']}/dataset_{task['repetition']:04d}.tar.gz"
    target = Path(root) / relative
    target.parent.mkdir(exist_ok=True)
    packed = compact_directory(base, target)
    shutil.rmtree(base)
    return {
        "n": task["n"],
        "repetition": task["repetition"],
        "status": record["status"],
        "verified": record["status"] == "complete" and verification_error is None,
        "verification_error": verification_error,
        **packed,
        "archive": relative,
        "manifest": relative + ".manifest.json",
    }


def replay_controls(config, *, power_study, output_dir, scratch, workers=1, limit=None):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four one-thread workers")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Positive replay limit required")
    # Read only the power ledger; the large-null cohort is not part of these replays.
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    power = Path(power_study).resolve()
    tasks = []
    model = read_json(power / "study.json")["calibration"]["model"]
    for row in map(json.loads, (power / "observations.jsonl").read_text().splitlines()):
        if row["cell"] in {"null_n64", "null_n256"}:
            tasks.append(
                {
                    "cohort": "power_controls",
                    "case": "independent_gaussian",
                    "n": row["cell_config"]["n_units"],
                    "repetition": row["repetition"],
                    "data_seed": row["data_seed"],
                    "source": str(power),
                    "ledger": row,
                    "model": model,
                }
            )
    if limit is not None:
        tasks = tasks[:limit]
    write_json(tasks, root / "private-source-plan.json")
    rows = []
    with (
        ProcessPoolExecutor(max_workers=workers) as executor,
        (root / "observations.jsonl").open("x") as handle,
    ):
        for row in executor.map(
            _replay,
            ((str(root), r, config.to_dict(), str(Path(scratch).resolve())) for r in tasks),
            chunksize=1,
        ):
            rows.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
    result = {
        "requested_replays": len(tasks),
        "verified_replays": sum(r["verified"] for r in rows),
        "successful_fits": sum(r["status"] == "complete" for r in rows),
        "failed_fits": sum(r["status"] != "complete" for r in rows),
        "calibration_draws": 0,
    }
    write_json(result, root / "summary.json")
    return result
