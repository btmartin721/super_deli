"""Bounded split/optimization diagnostics around an unchanged predictive selector."""

from __future__ import annotations

import contextlib
import json
import shutil
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from pathlib import Path

from super_deli.benchmarking.local_archives import compact_directory, write_progress
from super_deli.benchmarking.predictive_study import retained_bytes
from super_deli.core.provenance import configuration_hash
from super_deli.io.config import read_json, write_json
from super_deli.models.predictive_recovery_config import PredictiveRecoveryConfig
from super_deli.models.predictive_study_config import PredictiveCell

COUNT_KEYS = (
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


def fit_partition(base, values, layout, dataset, source, model, screen, template=None):
    from super_deli.workflows.predictive import finish_screen, run_predictive, summary_for

    workflow = base / "workflow"
    if template is None:
        return run_predictive(
            dataset, model_config=model, screen_config=screen, base_dir=source, output_dir=workflow
        )
    workflow.mkdir(parents=True)
    for name in ("full", "layout.json", "source_values.npz"):
        old = template / "workflow" / name
        new = workflow / name
        if old.is_dir():
            shutil.copytree(old, new)
        else:
            shutil.copy2(old, new)
    definition = read_json(template / "workflow/configuration.json")
    definition["screening"] = screen.to_dict()
    write_json(definition, workflow / "configuration.json")
    write_json(
        {"reused_full_search_from": str(template / "workflow/full"), "additional_full_fits": 0},
        workflow / "shared_full_search.json",
    )
    selection, records = finish_screen(
        workflow, values, layout, model, screen, read_json(workflow / "full/record.json")
    )
    return summary_for(workflow, selection, records, screen)


def count_unique(base, declaration):
    result = {k: declaration[k] for k in ("planned_searches", "planned_oracles")}
    records = []
    for p in base.glob("part_*/workflow/*/record.json"):
        if p.parent.name == "full" and (p.parent.parent / "shared_full_search.json").exists():
            continue
        records.append(read_json(p))
    result.update(
        attempted_searches=len(records),
        successful_searches=sum(r["status"] == "complete" for r in records),
        failed_searches=sum(r["status"] != "complete" for r in records),
        unattempted_searches=result["planned_searches"] - len(records),
    )
    records = [read_json(p) for p in base.glob("oracles/*/result.json")]
    result.update(
        attempted_oracles=len(records),
        successful_oracles=sum(r["status"] == "complete" for r in records),
        failed_oracles=sum(r["status"] != "complete" for r in records),
        unattempted_oracles=result["planned_oracles"] - len(records),
    )
    if any(v < 0 for v in result.values()):
        raise ValueError("Over-budget fit accounting")
    return result


def _observation(task):
    import numpy as np
    from threadpoolctl import threadpool_limits

    from super_deli.benchmarking.aic_diagnostic_simulation import heldout_layout
    from super_deli.benchmarking.predictive_recovery_metrics import (
        diagnose_workflow,
        oracle_diagnostics,
    )
    from super_deli.benchmarking.predictive_study import evaluate, simulate
    from super_deli.models.grouped_null import write_raw_views
    from super_deli.workflows.predictive import save_arrays

    source, raw_config, declaration = task
    root, config = Path(source), PredictiveRecoveryConfig.from_dict(raw_config)
    study = config.study
    cell = PredictiveCell.from_dict(declaration["cell_config"])
    if shutil.disk_usage(root).free < study.min_free_bytes + 128 * 1024**2:
        raise RuntimeError("Insufficient checked space; preserve partial evidence")
    base = root / ".working" / f"{cell.name}_{declaration['repetition']:04d}"
    base.mkdir(parents=True, exist_ok=False)
    write_json(declaration, base / "declaration.json")
    values, layout, truth = simulate(cell, cell.n_units, declaration["data_seed"])
    tests, test_layout, test_truth = simulate(cell, study.n_test, declaration["test_seed"])
    test_layout = heldout_layout(test_layout)
    write_json(layout, base / "layout.json")
    write_json(test_layout, base / "external_layout.json")
    save_arrays(base / "values.npz", **values)
    save_arrays(base / "external_values.npz", **tests)
    save_arrays(base / "truth.npz", training=truth, external=test_truth)
    dataset = write_raw_views(values, layout, base / "observed")
    parts = []
    errors = []
    definitions = [
        (f"part_{i:03d}", seed, study.model, i > 0)
        for i, seed in enumerate(declaration["partition_seeds"])
    ]
    definitions.append(
        (
            "part_extended",
            declaration["partition_seeds"][0],
            replace(study.model, n_init=config.extended_starts),
            False,
        )
    )
    with (
        threadpool_limits(limits=1),
        (base / "stdout.log").open("x") as out,
        (base / "stderr.log").open("x") as err,
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
    ):
        for name, seed, model, reuse in definitions:
            destination = base / name
            destination.mkdir()
            screen = replace(study.screening, random_seed=seed)
            try:
                if reuse and not (base / "part_000/workflow/full/run/gmm_candidates.npz").exists():
                    raise ValueError("Shared full search unavailable; do not replace it")
                fitted = fit_partition(
                    destination,
                    values,
                    layout,
                    dataset,
                    base / "observed",
                    model,
                    screen,
                    base / "part_000" if reuse else None,
                )
                performance = evaluate(
                    destination,
                    values,
                    layout,
                    truth,
                    cell,
                    tests,
                    test_layout,
                    fitted["selection"],
                )
                write_json(performance, destination / "performance.json")
                metrics = diagnose_workflow(
                    destination / "workflow",
                    truth if cell.role == "alternative" else None,
                    tests,
                    test_layout,
                    model,
                    destination / "diagnostics",
                )
                parts.append(
                    {
                        "name": name,
                        "screen_seed": seed,
                        "n_init": model.n_init,
                        "shared_full_search": reuse,
                        "selection": fitted["selection"],
                        "performance": performance,
                        "gates": metrics["gates"],
                    }
                )
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                write_json({"error": str(exc)}, destination / "diagnostic_failure.json")
                errors.append({"part": name, "error": str(exc)})
                parts.append(
                    {
                        "name": name,
                        "screen_seed": seed,
                        "n_init": model.n_init,
                        "shared_full_search": reuse,
                        "selection": {"status": "withheld", "screened_k": None},
                        "performance": {k: None for k in ("ordinary", "screened", "bic")},
                        "gates": None,
                    }
                )
        if cell.role == "alternative":
            oracle_diagnostics(
                base / "part_000/workflow", truth, tests, test_layout, study.model, base / "oracles"
            )
    record = {**declaration, **count_unique(base, declaration), "parts": parts, "errors": errors}
    write_json(record, base / "record.json")
    relative = f"{cell.name}/dataset_{declaration['repetition']:04d}.tar.gz"
    target = root / relative
    target.parent.mkdir(exist_ok=True)
    packed = compact_directory(base, target)
    shutil.rmtree(base)
    return {**record, **packed, "archive": relative, "manifest": relative + ".manifest.json"}


def run_study(config, *, output_dir, workers=1, preflight=None):
    config = PredictiveRecoveryConfig.from_dict(config.to_dict())
    root = Path(output_dir).resolve()
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("One to four one-thread workers required")
    plan = config.plan()
    count = sum(r["planned_searches"] + r["planned_oracles"] for r in plan)
    if count > 146 and not preflight:
        raise ValueError("Measured independently audited preflight required")
    checked = read_json(preflight) if preflight else None
    if checked and (
        checked["status"] != "passed"
        or checked["configuration_sha256"] != configuration_hash(config.to_dict())
    ):
        raise ValueError("Storage preflight differs from frozen study")
    artifact_root = Path(checked["artifact_root"]).resolve() if checked else root
    if checked and (root == artifact_root or not root.is_relative_to(artifact_root)):
        raise ValueError("Study must be inside the checked artifact root")
    root.mkdir(parents=True, exist_ok=False)
    write_json(config.to_dict(), root / "configuration.json")
    write_json(plan, root / "plan.json")
    records = []

    def progress(status):
        result = {
            "status": status,
            "observations": len(records),
            "requested_observations": len(plan),
            **{k: sum(r[k] for r in records) for k in COUNT_KEYS},
        }
        write_progress(result, root / "progress.json")
        return result

    with (
        ProcessPoolExecutor(max_workers=workers) as pool,
        (root / "observations.jsonl").open("x") as ledger,
    ):
        tasks = iter(plan)
        pending = {}

        def submit():
            row = next(tasks, None)
            if row is not None:
                pending[pool.submit(_observation, (str(root), config.to_dict(), row))] = row

        for _ in range(workers):
            submit()
        try:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    record = future.result()
                    records.append(record)
                    ledger.write(json.dumps(record, sort_keys=True) + "\n")
                    ledger.flush()
                    progress("running")
                    if retained_bytes(artifact_root) > config.study.max_artifact_bytes:
                        raise RuntimeError("Artifact cap reached; preserve all partial evidence")
                    submit()
        except BaseException as exc:
            for future in pending:
                future.cancel()
            write_json(
                {**progress("interrupted"), "error": str(exc), "pending": list(pending.values())},
                root / "execution_failure.json",
            )
            raise
    result = progress("complete")
    write_json(result, root / "summary.json")
    return result
