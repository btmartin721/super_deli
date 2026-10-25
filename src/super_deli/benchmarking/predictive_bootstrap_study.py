"""Bounded independent datasets, complete retained draws and lossless local archives."""

import contextlib
import json
import shutil
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from pathlib import Path

from super_deli.benchmarking.local_archives import compact_directory, write_progress
from super_deli.benchmarking.predictive_study import retained_bytes, simulate
from super_deli.core.provenance import configuration_hash
from super_deli.io.config import read_json, write_json
from super_deli.models.predictive_bootstrap_config import PredictiveBootstrapConfig
from super_deli.models.predictive_bootstrap_study import PredictiveBootstrapStudy
from super_deli.models.predictive_study_config import PredictiveCell

COUNTS = ("attempted_fits", "successful_fits", "failed_fits", "unattempted_fits")


def _observation(task):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    from super_deli.models.grouped_null import write_raw_views
    from super_deli.workflows.predictive import save_arrays
    from super_deli.workflows.predictive_bootstrap import run_predictive_bootstrap

    source, raw, declaration = task
    root, c = Path(source), PredictiveBootstrapStudy.from_dict(raw)
    if shutil.disk_usage(root).free < c.min_free_bytes + 128 * 1024**2:
        raise RuntimeError("Insufficient checked space; retain partial evidence")
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
        write_json(layout, base / "generating_layout.json")
        dataset = write_raw_views(values, layout, base / "observed")
        result = run_predictive_bootstrap(
            dataset,
            model_config=c.model,
            screen_config=replace(c.screening, random_seed=declaration["split_seed"]),
            bootstrap_config=PredictiveBootstrapConfig(
                c.n_bootstrap, declaration["bootstrap_seed"]
            ),
            base_dir=base / "observed",
            output_dir=base / "workflow",
        )
    k = result["point"]["selection"]["screened_k"]
    point_ari = None
    group_truth = cell.law in {"complementary", "unequal_variance"}
    if k is not None:
        with np.load(base / "workflow/point/selected_memberships.npz", allow_pickle=False) as p:
            if group_truth:
                point_ari = float(adjusted_rand_score(truth, p["labels"]))
    performance = [
        {
            "replicate": row["replicate"],
            "selected_k": row.get("selected_k"),
            "ari": float(adjusted_rand_score(truth, row["labels"]))
            if group_truth and row["status"] == "complete"
            else None,
        }
        for row in result["replicates"]
    ]
    write_json(performance, base / "truth_evaluation.json")
    record = {
        **declaration,
        **{k: result[k] for k in COUNTS},
        "point_selection": result["point"]["selection"],
        "point_ari": point_ari,
        "stability": result.get("stability"),
        "failed_draws": sum(r["status"] != "complete" for r in result["replicates"]),
        "successful_draws": sum(r["status"] == "complete" for r in result["replicates"]),
        "unattempted_draws": c.n_bootstrap - len(result["replicates"]),
    }
    write_json(record, base / "record.json")
    relative = f"{cell.name}/dataset_{declaration['repetition']:04d}.tar.gz"
    packed = compact_directory(base, root / relative)
    shutil.rmtree(base)
    return {**record, **packed, "archive": relative, "manifest": relative + ".manifest.json"}


def run_study(config, *, output_dir, workers=1, preflight=None):
    c = PredictiveBootstrapStudy.from_dict(config.to_dict())
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
            raise ValueError("Storage gate/configuration/output mismatch")
    else:
        root = output.parent
    if shutil.disk_usage(output.parent).free < c.min_free_bytes + 128 * 1024**2:
        raise RuntimeError("Insufficient checked free space")
    output.mkdir(parents=True, exist_ok=False)
    write_json(c.to_dict(), output / "configuration.json")
    plan = c.plan()
    write_json(plan, output / "plan.json")
    rows = []

    def report(status):
        result = {
            "status": status,
            "observations": len(rows),
            "requested": len(plan),
            "planned_fits": sum(r["planned_fits"] for r in plan),
            **{k: sum(r[k] for r in rows) for k in COUNTS},
            **{
                k: sum(r[k] for r in rows)
                for k in ("failed_draws", "successful_draws", "unattempted_draws")
            },
        }
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
                        raise RuntimeError(
                            "Runtime storage guard failed; preserve partial evidence"
                        )
                    submit()
        except BaseException as exc:
            for future in pending:
                future.cancel()
            write_json(
                {**report("interrupted"), "error": str(exc)}, output / "execution_failure.json"
            )
            raise
    result = report("complete")
    write_json(result, output / "summary.json")
    return result


def observation(task):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _observation(task)
