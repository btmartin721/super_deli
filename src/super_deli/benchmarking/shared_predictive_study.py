"""Paired full/tied raw-pipeline runs with verified local archive retention."""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from super_deli.benchmarking.local_archives import compact_directory, materialize, write_progress
from super_deli.benchmarking.predictive_study import retained_bytes
from super_deli.core.provenance import configuration_hash
from super_deli.io.config import read_json, write_json
from super_deli.models.shared_predictive_config import SharedPredictiveConfig

COUNTS = ("attempted_fits", "successful_fits", "failed_fits", "unattempted_fits")


def same_maps(left, right):
    import numpy as np

    compared = 0
    for prefix in ["full"] + [f"fold_{j:03d}" for j in range(5)]:
        a, b = left / "workflow" / prefix / "run", right / "workflow" / prefix / "run"
        if (
            not a.exists()
            or not b.exists()
            or not (a / "gmm_candidates.npz").exists()
            or not (b / "gmm_candidates.npz").exists()
        ):
            continue
        compared += 1
        for relative in (
            "fusion.npz",
            "view_000/source_values.npy",
            "view_001/source_values.npy",
            "view_000/parameters/pca.npz",
            "view_001/parameters/pca.npz",
            "view_000/parameters/profiles.npz",
            "view_001/parameters/profiles.npz",
        ):
            if relative.endswith(".npy"):
                np.testing.assert_array_equal(
                    np.load(a / relative, allow_pickle=False),
                    np.load(b / relative, allow_pickle=False),
                )
            else:
                with (
                    np.load(a / relative, allow_pickle=False) as x,
                    np.load(b / relative, allow_pickle=False) as y,
                ):
                    if set(x.files) != set(y.files):
                        raise ValueError("Paired transformation schemas differ")
                    for key in x.files:
                        np.testing.assert_array_equal(x[key], y[key])
        with (
            np.load(a / "gmm_candidates.npz", allow_pickle=False) as x,
            np.load(b / "gmm_candidates.npz", allow_pickle=False) as y,
        ):
            np.testing.assert_array_equal(x["scores"], y["scores"])
        for i in range(2):
            if read_json(a / f"view_{i:03d}/source_rows.json") != read_json(
                b / f"view_{i:03d}/source_rows.json"
            ):
                raise ValueError("Paired raw identities differ")
    return compared


def family_run(root, config, declaration):
    """Reuse production fits; add fixed external evaluation after all selection."""
    import contextlib
    from dataclasses import replace
    from time import perf_counter

    import numpy as np

    from super_deli.benchmarking.aic_diagnostic_simulation import heldout_layout
    from super_deli.benchmarking.predictive_study import evaluate, simulate
    from super_deli.io.predictive_inputs import project_exported
    from super_deli.models.grouped_null import write_raw_views
    from super_deli.models.predictive_study_config import PredictiveCell
    from super_deli.workflows.predictive import predict_pool, run_predictive, save_arrays

    cell = PredictiveCell.from_dict(declaration["cell_config"])
    base = root / ".working" / f"{cell.name}_{declaration['repetition']:04d}"
    base.mkdir(parents=True, exist_ok=False)
    write_json(declaration, base / "declaration.json")
    screening = replace(config.screening, random_seed=declaration["split_seed"])
    start = perf_counter()
    with (
        (base / "stdout.log").open("x") as out,
        (base / "stderr.log").open("x") as err,
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
    ):
        values, layout, truth = simulate(cell, cell.n_units, declaration["data_seed"])
        tests, test_layout, test_truth = simulate(cell, config.n_test, declaration["test_seed"])
        test_layout = heldout_layout(test_layout)
        write_json(layout, base / "generating_layout.json")
        write_json(test_layout, base / "external_layout.json")
        save_arrays(base / "values.npz", **values)
        save_arrays(base / "external_values.npz", **tests)
        save_arrays(base / "truth.npz", training=truth, external=test_truth)
        dataset = write_raw_views(values, layout, base / "observed")
        try:
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
            records = [read_json(p) for p in (base / "workflow").glob("*/record.json")]
            result = {
                "selection": {"status": "withheld", "screened_k": None, "error": str(exc)},
                "attempted_fits": len(records),
                "successful_fits": sum(r["status"] == "complete" for r in records),
                "failed_fits": sum(r["status"] != "complete" for r in records),
                "unattempted_fits": declaration["planned_fits"] - len(records),
            }
            performance = {name: None for name in ("ordinary", "screened", "bic")}
            write_json({"error": str(exc)}, base / "failure.json")
        external = []
        for i in range(5):
            prefix = f"fold_{i:03d}"
            run = base / "workflow" / prefix / "run"
            if not (run / "gmm_candidates.npz").exists():
                external.append({"prefix": prefix, "status": "unavailable"})
                continue
            try:
                scores = project_exported(run, tests, test_layout)
                logs = predict_pool(run, scores, base / (prefix + "_external.npz"))
                external.append(
                    {
                        "prefix": prefix,
                        "status": "complete",
                        "gains": {
                            k: float(np.mean(np.asarray(v) - np.asarray(logs["1"])))
                            for k, v in logs.items()
                        },
                    }
                )
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                external.append({"prefix": prefix, "status": "failed", "error": str(exc)})
        write_json(external, base / "external_folds.json")
    record = {
        **declaration,
        **{k: result[k] for k in COUNTS},
        "selection": result["selection"],
        "performance": performance,
        "elapsed_seconds": perf_counter() - start,
        "external_fold_failures": sum(r["status"] == "failed" for r in external),
    }
    write_json(record, base / "record.json")
    relative = f"{cell.name}/dataset_{declaration['repetition']:04d}.tar.gz"
    packed = compact_directory(base, root / relative)
    shutil.rmtree(base)
    return {**record, **packed, "archive": relative, "manifest": relative + ".manifest.json"}


def observation(task):
    from threadpoolctl import threadpool_limits

    root, raw, declaration = task
    c = SharedPredictiveConfig.from_dict(raw)
    if shutil.disk_usage(root).free < c.study.min_free_bytes + 128 * 1024**2:
        raise RuntimeError("Insufficient checked storage; preserve partial evidence")
    base = Path(root) / ".working" / f"{declaration['cell']}_{declaration['repetition']:04d}"
    base.mkdir(parents=True, exist_ok=False)
    write_json(declaration, base / "declaration.json")
    arms = {}
    with threadpool_limits(limits=1):
        for family in ("full", "tied"):
            source = base / family
            source.mkdir()
            row = {**declaration, "planned_fits": declaration["planned_fits"] // 2}
            arms[family] = family_run(source, c.family(family), row)
        with (
            materialize(
                base / "full" / arms["full"]["archive"], arms["full"], scratch=base / ".scratch"
            ) as a,
            materialize(
                base / "tied" / arms["tied"]["archive"], arms["tied"], scratch=base / ".scratch"
            ) as b,
        ):
            compared = same_maps(a, b)
    result = {
        **declaration,
        "arms": arms,
        **{k: sum(r[k] for r in arms.values()) for k in COUNTS},
        "maps_verified_identical": compared == 6,
        "map_comparisons": compared,
        "external_fold_failures": sum(r["external_fold_failures"] for r in arms.values()),
        "screening_withheld": sum(r["selection"]["status"] != "complete" for r in arms.values()),
    }
    write_json(result, base / "record.json")
    relative = f"{declaration['cell']}/dataset_{declaration['repetition']:04d}.tar.gz"
    packed = compact_directory(base, Path(root) / relative)
    shutil.rmtree(base)
    return {**result, **packed, "archive": relative, "manifest": relative + ".manifest.json"}


def run_study(config, *, output_dir, workers=1, preflight=None):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four single-thread workers")
    c = SharedPredictiveConfig.from_dict(config.to_dict())
    plan = c.plan()
    output = Path(output_dir).resolve()
    if sum(r["planned_fits"] for r in plan) > 168:
        if preflight is None:
            raise ValueError("Measured storage preflight required")
        gate = read_json(preflight)
        base = Path(gate["artifact_root"])
        if (
            gate["status"] != "passed"
            or gate["configuration_sha256"] != configuration_hash(c.to_dict())
            or not output.is_relative_to(base)
        ):
            raise ValueError("Measured gate/configuration/output mismatch")
    else:
        base = output.parent
    if shutil.disk_usage(output.parent).free < c.study.min_free_bytes + 128 * 1024**2:
        raise RuntimeError("Insufficient checked free space")
    output.mkdir(parents=True, exist_ok=False)
    write_json(c.to_dict(), output / "configuration.json")
    write_json(plan, output / "plan.json")
    rows = []
    with (
        ProcessPoolExecutor(max_workers=workers) as pool,
        (output / "observations.jsonl").open("x") as handle,
    ):
        for row in pool.map(
            observation, ((str(output), c.to_dict(), r) for r in plan), chunksize=1
        ):
            rows.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            result = {
                "status": "running",
                "observations": len(rows),
                "requested": len(plan),
                **{k: sum(r[k] for r in rows) for k in COUNTS},
            }
            write_progress(result, output / "progress.json")
            if (
                retained_bytes(base) > c.study.max_artifact_bytes
                or shutil.disk_usage(base).free < c.study.min_free_bytes
            ):
                raise RuntimeError("Runtime storage guard failed; preserve partial evidence")
    result.update(
        status="complete",
        screening_withheld=sum(r["screening_withheld"] for r in rows),
        external_fold_failures=sum(r["external_fold_failures"] for r in rows),
    )
    write_json(result, output / "summary.json")
    write_progress(result, output / "progress.json")
    return result
