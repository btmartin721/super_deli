"""Frozen paired ordinary/screened AIC validation with local lossless evidence."""

from __future__ import annotations

import contextlib
import json
import shutil
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from super_deli.benchmarking.local_archives import compact_directory, write_progress
from super_deli.core.provenance import configuration_hash
from super_deli.io.config import read_json, write_json
from super_deli.models.predictive_study_config import PredictiveCell, PredictiveStudyConfig


def retained_bytes(root):
    """Count retained evidence while other workers atomically publish archives."""
    root = Path(root)
    total = 0
    for path in root.rglob("*"):
        if {".working", ".scratch"}.intersection(path.relative_to(root).parts):
            continue
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            # A verified .partial archive may have been renamed since traversal.
            continue
    return total


def simulate(cell, n, seed):
    import numpy as np

    from super_deli.benchmarking.grouped_pca_study import simulate as stress
    from super_deli.benchmarking.grouped_power_study import simulate as power
    from super_deli.benchmarking.grouped_study import simulate_grouped
    from super_deli.models.grouped_power_config import PowerCell

    if cell.law == "gaussian_scores":
        return np.random.default_rng(seed).standard_normal((n, 4)) / 2, None, None
    if cell.law in {"cline", "unequal_variance"}:
        return stress(cell.law, n, seed, target_level="locality")
    if cell.law == "complementary":
        return power(PowerCell(cell.name, cell.law, n, 1, cell.signal), seed)
    return simulate_grouped(cell.law, n, seed)


def evaluate(base, values, layout, truth, cell, tests, test_layout, selection):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    from super_deli.io.predictive_inputs import project_exported
    from super_deli.models.predictive_selection import gaussian_predictions
    from super_deli.workflows.predictive import predict_pool

    run = base / "workflow/full/run"
    if not (run / "gmm_candidates.npz").exists():
        return {name: None for name in ("ordinary", "screened", "bic")}
    scores = tests if cell.law == "gaussian_scores" else project_exported(run, tests, test_layout)
    logs = predict_pool(run, scores, base / "external_predictions.npz")
    result = {}
    with np.load(run / "gmm_candidates.npz", allow_pickle=False) as pool:
        for name, key in (
            ("ordinary", "ordinary_aic_k"),
            ("screened", "screened_k"),
            ("bic", "descriptive_bic_k"),
        ):
            k = selection.get(key)
            if k is None:
                result[name] = None
                continue
            gain = np.asarray(logs[str(k)]) - np.asarray(logs["1"])
            labels = pool[f"k{k}_labels"]
            _, prob = gaussian_predictions(pool["scores"], pool, k)
            result[name] = {
                "k": k,
                "ari": float(adjusted_rand_score(truth, labels))
                if cell.law in {"complementary", "unequal_variance"}
                else None,
                "external_mean_logdensity_gain": float(gain.mean()),
                "external_mcse_conditional_on_training": float(
                    gain.std(ddof=1) / np.sqrt(len(gain))
                ),
                "mean_assignment_entropy": float(
                    -np.sum(prob * np.log(np.maximum(prob, 1e-300))) / len(prob)
                ),
            }
    return result


def _observation(task):
    import numpy as np
    from threadpoolctl import threadpool_limits

    from super_deli.benchmarking.aic_diagnostic_simulation import heldout_layout
    from super_deli.models.grouped_null import write_raw_views
    from super_deli.workflows.predictive import run_predictive, run_score_screen, save_arrays

    source, raw_config, declaration = task
    root, config = Path(source), PredictiveStudyConfig.from_dict(raw_config)
    cell = PredictiveCell.from_dict(declaration["cell_config"])
    if shutil.disk_usage(root).free < config.min_free_bytes + 128 * 1024**2:
        raise RuntimeError("Insufficient checked working space; retain partial evidence")
    base = root / ".working" / f"{cell.name}_{declaration['repetition']:04d}"
    base.mkdir(parents=True, exist_ok=False)
    write_json(declaration, base / "declaration.json")
    screening = replace(config.screening, random_seed=declaration["split_seed"])
    start = perf_counter()
    with (
        threadpool_limits(limits=1),
        (base / "stdout.log").open("x") as out,
        (base / "stderr.log").open("x") as err,
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
    ):
        values, layout, truth = simulate(cell, cell.n_units, declaration["data_seed"])
        tests, test_layout, test_truth = simulate(cell, config.n_test, declaration["test_seed"])
        direct = cell.law == "gaussian_scores"
        if direct:
            save_arrays(base / "inputs.npz", training=values, external=tests)
        else:
            test_layout = heldout_layout(test_layout)
            write_json(layout, base / "generating_layout.json")
            write_json(test_layout, base / "external_layout.json")
            save_arrays(base / "values.npz", **values)
            save_arrays(base / "external_values.npz", **tests)
            save_arrays(base / "truth.npz", training=truth, external=test_truth)
            dataset = write_raw_views(values, layout, base / "observed")
        try:
            if direct:
                result = run_score_screen(
                    values,
                    model=config.model,
                    screen_config=screening,
                    output_dir=base / "workflow",
                )
            else:
                result = run_predictive(
                    dataset,
                    model_config=config.model,
                    screen_config=screening,
                    base_dir=base / "observed",
                    output_dir=base / "workflow",
                )
            performance = evaluate(
                base, values, layout, truth, cell, tests, test_layout, result["selection"]
            )
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            records = [read_json(p) for p in sorted((base / "workflow").glob("*/record.json"))]
            result = {
                "selection": {"status": "withheld", "screened_k": None, "error": str(exc)},
                "attempted_fits": len(records),
                "successful_fits": sum(r["status"] == "complete" for r in records),
                "failed_fits": sum(r["status"] != "complete" for r in records),
                "unattempted_fits": declaration["planned_fits"] - len(records),
            }
            performance = {name: None for name in ("ordinary", "screened", "bic")}
            write_json({"error": str(exc)}, base / "failure.json")
    record = {
        **declaration,
        **{
            k: result[k]
            for k in ("attempted_fits", "successful_fits", "failed_fits", "unattempted_fits")
        },
        "selection": result["selection"],
        "performance": performance,
        "elapsed_seconds": perf_counter() - start,
    }
    write_json(record, base / "record.json")
    relative = f"{cell.name}/dataset_{declaration['repetition']:04d}.tar.gz"
    target = root / relative
    target.parent.mkdir(exist_ok=True)
    packed = compact_directory(base, target)
    shutil.rmtree(base)
    return {**record, **packed, "archive": relative, "manifest": relative + ".manifest.json"}


def run_study(config, *, output_dir, workers=1, preflight=None):
    config = PredictiveStudyConfig.from_dict(config.to_dict())
    root = Path(output_dir).resolve()
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four one-thread workers")
    plan = config.plan()
    if sum(r["planned_fits"] for r in plan) > 102 and not preflight:
        raise ValueError("Measured storage preflight required before full study")
    checked = read_json(preflight) if preflight else None
    if checked and (
        checked["status"] != "passed"
        or checked["configuration_sha256"] != configuration_hash(config.to_dict())
    ):
        raise ValueError("Storage preflight differs from frozen configuration")
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
            "planned_fits": sum(r["planned_fits"] for r in plan),
            **{
                k: sum(r[k] for r in records)
                for k in ("attempted_fits", "successful_fits", "failed_fits", "unattempted_fits")
            },
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
                    if retained_bytes(artifact_root) > config.max_artifact_bytes:
                        raise RuntimeError("Artifact cap reached; preserve partial evidence")
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
