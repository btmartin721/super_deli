"""Prespecified paired mechanisms, with complete immutable evidence and separate oracles."""

import contextlib
import json
import shutil
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from pathlib import Path

from super_deli.benchmarking.bootstrap_diagnostic_gates import unique_support
from super_deli.benchmarking.bootstrap_diagnostic_metrics import (
    diagnose_workflow,
    truth_diagnostics,
)
from super_deli.benchmarking.local_archives import compact_directory, write_progress
from super_deli.benchmarking.predictive_study import retained_bytes, simulate
from super_deli.core.provenance import configuration_hash
from super_deli.io.config import read_json, write_json
from super_deli.models.bootstrap_diagnostic_config import BootstrapDiagnosticConfig
from super_deli.models.predictive_study_config import PredictiveCell

COUNTS = ("attempted_fits", "successful_fits", "failed_fits", "unattempted_fits")


def execute(callback, directory):
    import numpy as np

    directory = Path(directory)
    try:
        return callback()
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError, OSError) as exc:
        directory.mkdir(parents=True, exist_ok=True)
        records = [read_json(p) for p in directory.glob("*/record.json")]
        summary = dict(
            selection=dict(status="withheld", screened_k=None, error=str(exc)),
            planned_fits=6,
            attempted_fits=len(records),
            successful_fits=sum(r["status"] == "complete" for r in records),
            failed_fits=sum(r["status"] != "complete" for r in records),
            unattempted_fits=6 - len(records),
        )
        write_json(
            dict(status="failed", error=str(exc)), directory / "diagnostic_execution_failure.json"
        )
        if not (directory / "summary.json").exists():
            write_json(summary, directory / "summary.json")
        return summary


def _observation(task):
    import numpy as np

    from super_deli.models.grouped_null import write_raw_views
    from super_deli.workflows.predictive import run_predictive, save_arrays
    from super_deli.workflows.predictive_resampling import occurrence_screen

    source, raw, declaration = task
    root, c = Path(source), BootstrapDiagnosticConfig.from_dict(raw)
    if shutil.disk_usage(root).free < c.min_free_bytes + 128 * 1024**2:
        raise RuntimeError("Insufficient checked scratch space; preserve partial evidence")
    cell = PredictiveCell.from_dict(declaration["cell_config"])
    base = root / ".working" / f"{cell.name}_{declaration['repetition']:04d}"
    base.mkdir(parents=True, exist_ok=False)
    write_json(declaration, base / "declaration.json")
    with (
        (base / "stdout.log").open("x") as out,
        (base / "stderr.log").open("x") as err,
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
    ):
        values, layout, truth = simulate(cell, cell.n_units, declaration["data_seed"])
        save_arrays(base / "values.npz", **values)
        save_arrays(base / "truth.npz", labels=truth)
        write_json(layout, base / "layout.json")
        dataset = write_raw_views(values, layout, base / "observed")
        screen = replace(c.screening, random_seed=declaration["split_seed"])
        for name in ("baseline", "unique", "extended", "diagnostics"):
            (base / name).mkdir()
        models = dict(
            baseline=c.model, unique=c.model, extended=replace(c.model, n_init=c.extended_starts)
        )
        workflows = []
        for arm in ("baseline", "extended"):
            target = base / arm / "point"
            summary = execute(
                lambda arm=arm, target=target: run_predictive(
                    dataset,
                    model_config=models[arm],
                    screen_config=screen,
                    base_dir=base / "observed",
                    output_dir=target,
                ),
                target,
            )
            workflows.append(
                dict(
                    arm=arm,
                    replicate=-1,
                    prefix=f"{arm}/point",
                    indices=list(range(cell.n_units)),
                    **{k: summary[k] for k in COUNTS},
                )
            )
        assignments = (
            read_json(base / "baseline/point/fold_assignments.json")
            if (base / "baseline/point/fold_assignments.json").exists()
            else []
        )
        write_json(assignments, base / "fold_assignments.json")
        draws = []
        for i, seed in enumerate(c.draw_seeds(declaration)):
            draw = np.random.default_rng(seed).integers(cell.n_units, size=cell.n_units)
            support = np.asarray(unique_support(draw), dtype=int)
            row = dict(replicate=i, seed=seed, indices=draw.tolist(), support=support.tolist())
            draws.append(row)
            write_json(row, base / f"draw_{i:04d}.json")
            for arm, indices in [("baseline", draw), ("unique", support)] + (
                [("extended", draw)] if i < c.extended_draws else []
            ):
                prefix = f"{arm}/draw_{i:04d}"
                target = base / prefix
                if assignments:
                    summary = execute(
                        lambda arm=arm, target=target, indices=indices: occurrence_screen(
                            target, values, layout, indices, models[arm], assignments
                        ),
                        target,
                    )
                else:

                    def unavailable():
                        raise ValueError("Original point fold partition unavailable")

                    summary = execute(unavailable, target)
                workflows.append(
                    dict(
                        arm=arm,
                        replicate=i,
                        prefix=prefix,
                        indices=indices.tolist(),
                        **{k: summary[k] for k in COUNTS},
                    )
                )
        # Labels enter only now, after every ordinary search in all arms is frozen.
        evaluated = truth if cell.role == "alternative" else None
        point_labels = None
        if (base / "baseline/point/selected_memberships.npz").exists():
            with np.load(base / "baseline/point/selected_memberships.npz", allow_pickle=False) as p:
                point_labels = p["labels"].copy()
        for row in workflows:
            target = base / "diagnostics" / row["arm"]
            target.mkdir(exist_ok=True)
            try:
                metrics = diagnose_workflow(
                    base / row["prefix"],
                    values,
                    layout,
                    row["indices"],
                    models[row["arm"]],
                    evaluated,
                    point_labels,
                )
            except (
                ValueError,
                RuntimeError,
                FloatingPointError,
                np.linalg.LinAlgError,
                OSError,
            ) as exc:
                metrics = dict(status="unavailable", error=str(exc))
            write_json(
                metrics,
                target
                / (
                    ("point" if row["replicate"] == -1 else f"draw_{row['replicate']:04d}")
                    + ".json"
                ),
            )
            row.update(
                selection=read_json(base / row["prefix"] / "summary.json")["selection"],
                diagnostic_available=metrics.get("gate", {}).get("status") == "available",
            )
        oracles = []
        if evaluated is not None:
            (base / "oracles").mkdir()
            for row in workflows:
                if row["arm"] == "baseline" and row["replicate"] < c.extended_draws:
                    name = "point" if row["replicate"] == -1 else f"draw_{row['replicate']:04d}"
                    entries = truth_diagnostics(
                        base / row["prefix"],
                        truth,
                        row["indices"],
                        c.model,
                        assignments,
                        base / "oracles" / name,
                    )
                    oracles.extend([dict(workflow=name, **v) for v in entries])
        write_json(workflows, base / "workflows.json")
        write_json(draws, base / "draws.json")
        write_json(oracles, base / "oracles.json")
    record = dict(
        declaration,
        **{k: sum(v[k] for v in workflows) for k in COUNTS},
        workflow_count=len(workflows),
        withheld_workflows=sum(v["selection"]["status"] != "complete" for v in workflows),
        unavailable_diagnostics=sum(not v["diagnostic_available"] for v in workflows),
        attempted_oracles=sum(v["attempted"] for v in oracles),
        successful_oracles=sum(v["status"] == "complete" for v in oracles),
        failed_oracles=sum(v["status"] == "failed" for v in oracles),
        unattempted_oracles=sum(v["status"] == "unattempted" for v in oracles),
    )
    write_json(record, base / "record.json")
    archive = f"{cell.name}/dataset_{declaration['repetition']:04d}.tar.gz"
    packed = compact_directory(base, root / archive)
    shutil.rmtree(base)
    return {**record, **packed, "archive": archive, "manifest": archive + ".manifest.json"}


def observation(task):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _observation(task)


def run_study(config, *, output_dir, workers=1, preflight=None):
    c = BootstrapDiagnosticConfig.from_dict(config.to_dict())
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four single-thread workers")
    output = Path(output_dir).resolve()
    if any(cell.n_datasets > 1 for cell in c.cells):
        if preflight is None:
            raise ValueError("Measured audited storage preflight required")
        gate = read_json(preflight)
        root = Path(gate["artifact_root"]).resolve()
        if (
            gate["status"] != "passed"
            or gate["configuration_sha256"] != configuration_hash(c.to_dict())
            or output == root
            or not output.is_relative_to(root)
        ):
            raise ValueError("Preflight/configuration/output mismatch")
    else:
        root = output.parent
    if shutil.disk_usage(output.parent).free < c.min_free_bytes + 128 * 1024**2:
        raise RuntimeError("Insufficient checked space")
    output.mkdir(parents=True, exist_ok=False)
    write_json(c.to_dict(), output / "configuration.json")
    plan = c.plan()
    write_json(plan, output / "plan.json")
    rows = []

    def report(status):
        result = dict(
            status=status,
            observations=len(rows),
            requested=len(plan),
            planned_searches=sum(r["planned_searches"] for r in plan),
            planned_oracles=sum(r["planned_oracles"] for r in plan),
            **{
                k: sum(r[k] for r in rows)
                for k in (
                    *COUNTS,
                    "attempted_oracles",
                    "successful_oracles",
                    "failed_oracles",
                    "unattempted_oracles",
                    "withheld_workflows",
                    "unavailable_diagnostics",
                )
            },
        )
        write_progress(result, output / "progress.json")
        return result

    with (
        ProcessPoolExecutor(max_workers=workers) as pool,
        (output / "observations.jsonl").open("x") as ledger,
    ):
        tasks = iter(plan)
        pending = set()

        def submit():
            row = next(tasks, None)
            if row is not None:
                pending.add(pool.submit(observation, (str(output), c.to_dict(), row)))

        for _ in range(workers):
            submit()
        try:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    row = future.result()
                    rows.append(row)
                    ledger.write(json.dumps(row, sort_keys=True) + "\n")
                    ledger.flush()
                    report("running")
                    if (
                        retained_bytes(root) > c.max_artifact_bytes
                        or shutil.disk_usage(root).free < c.min_free_bytes
                    ):
                        raise RuntimeError("Runtime storage guard failed; retain partial evidence")
                    submit()
        except BaseException as exc:
            for future in pending:
                future.cancel()
            write_json(
                dict(report("interrupted"), error=str(exc)), output / "execution_failure.json"
            )
            raise
    result = report("complete")
    write_json(result, output / "summary.json")
    return result
