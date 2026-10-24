"""Point-only paired Gaussian controls; immutable compact local evidence."""

from __future__ import annotations

import contextlib
import json
import shutil
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from time import perf_counter

from super_deli.benchmarking.aic_diagnostic_metrics import candidate_metrics
from super_deli.benchmarking.aic_diagnostic_simulation import (
    decompose,
    heldout_layout,
    homogeneous,
    project_saved,
)
from super_deli.benchmarking.local_archives import compact_directory, retained_bytes, write_progress
from super_deli.core.provenance import configuration_hash
from super_deli.io.config import read_json, write_json
from super_deli.models.aic_diagnostic_config import ARMS, AICDiagnosticConfig


def save_arrays(path, **values):
    import numpy as np

    with Path(path).open("xb") as handle:
        np.savez(handle, **values)


def evaluate(run, test_scores, counts, model, directory):
    metrics, held = candidate_metrics(
        (run / "gmm_candidates.npz").read_bytes(),
        read_json(run / "candidates.json"),
        test_scores=test_scores,
        counts=counts,
        reg_covar=model.reg_covar,
    )
    save_arrays(
        directory / "heldout.npz", scores=test_scores, **{"k" + k: v for k, v in held.items()}
    )
    write_json(metrics, directory / "metrics.json")
    return metrics


def fit_raw(base, name, values, layout, test_values, test_layout, config):
    import numpy as np

    from super_deli.models.grouped_null import write_raw_views
    from super_deli.workflows.grouped_likelihood import point_fitter
    from super_deli.workflows.integrative import run_integrative

    directory = base / name
    directory.mkdir()
    model = config.arm(name)
    write_json(model.to_dict(), directory / "model.json")
    dataset = write_raw_views(values, layout, directory / "raw")
    record = point_fitter(
        directory, dataset, directory / "raw", name, model=model, runner=run_integrative
    )
    write_json(record, directory / "record.json")
    if record["status"] == "complete":
        test_scores = project_saved(directory / "run", test_values, test_layout)
        metrics = evaluate(
            directory / "run", test_scores, np.asarray(layout["counts"]), model, directory
        )
        return {
            "arm": name,
            "status": "complete",
            "aic_k": metrics["aic_k"],
            "bic_k": metrics["bic_k"],
        }
    return {"arm": name, **record, "aic_k": None, "bic_k": None}


def _observation(task):
    import numpy as np
    from threadpoolctl import threadpool_limits

    source, raw_config, declaration = task
    root, config = Path(source), AICDiagnosticConfig.from_dict(raw_config)
    if shutil.disk_usage(root).free < config.min_free_bytes + 128 * 1024**2:
        raise RuntimeError("Insufficient working space; preserve partial evidence")
    base = root / ".working" / f"{declaration['cell']}_{declaration['repetition']:04d}"
    base.mkdir(parents=True)
    write_json(declaration, base / "declaration.json")
    start = perf_counter()
    rows = []
    with (
        threadpool_limits(limits=1),
        (base / "stdout.log").open("x") as out,
        (base / "stderr.log").open("x") as err,
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
    ):
        if declaration["case"] == "gaussian_scores":
            from super_deli.models.mixture import select_gaussian_mixture

            train = (
                np.random.default_rng(declaration["data_seed"]).standard_normal(
                    (declaration["n"], 4)
                )
                / 2
            )
            test = (
                np.random.default_rng(declaration["test_seed"]).standard_normal((config.n_test, 4))
                / 2
            )
            save_arrays(base / "input.npz", training=train, heldout=test)
            point_seed = int(
                np.random.SeedSequence(config.model.random_seed).spawn(1)[0].generate_state(1)[0]
            )
            try:
                fit = select_gaussian_mixture(
                    train,
                    config.model,
                    point_seed,
                    covariance_type="full",
                    unit_indices=np.arange(len(train)),
                )
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                rows.append(
                    {
                        "arm": "scores",
                        "status": "failed",
                        "error": str(exc),
                        "aic_k": None,
                        "bic_k": None,
                    }
                )
            else:
                run = base / "scores"
                run.mkdir()
                save_arrays(run / "gmm_candidates.npz", scores=train, **fit.candidate_arrays)
                write_json(fit.candidates, run / "candidates.json")
                metrics = evaluate(run, test, None, config.model, run)
                rows.append(
                    {
                        "arm": "scores",
                        "status": "complete",
                        "point_seed": point_seed,
                        "aic_k": metrics["aic_k"],
                        "bic_k": metrics["bic_k"],
                    }
                )
        else:
            values, layout, latent, residual = decompose(
                declaration["case"], declaration["n"], declaration["data_seed"]
            )
            tests, raw_test_layout, test_latent, test_residual = decompose(
                declaration["case"], config.n_test, declaration["test_seed"]
            )
            test_layout = heldout_layout(raw_test_layout)
            r = float(np.mean(1 / np.asarray(layout["counts"])))
            write_json(layout, base / "layout.json")
            write_json(test_layout, base / "heldout_layout.json")
            write_json(
                {
                    "reciprocal_count_mean": r,
                    "scope": (
                        "Deliberately changed simulation residual law; not empirical correction"
                    ),
                },
                base / "intervention.json",
            )
            save_arrays(base / "training.npz", **values, latent=latent, residual=residual)
            save_arrays(
                base / "heldout_input.npz", **tests, latent=test_latent, residual=test_residual
            )
            for name in ARMS:
                train, test = values, tests
                if name == "homogeneous_mean_noise":
                    train = homogeneous(values, layout, latent, residual, r)
                    test = homogeneous(tests, test_layout, test_latent, test_residual, r)
                rows.append(fit_raw(base, name, train, layout, test, test_layout, config))
    success = sum(r["status"] == "complete" for r in rows)
    record = {
        **declaration,
        "points": rows,
        "attempted_fits": len(rows),
        "successful_fits": success,
        "failed_fits": len(rows) - success,
        "unattempted_fits": declaration["planned_fits"] - len(rows),
        "elapsed_seconds": perf_counter() - start,
    }
    write_json(record, base / "record.json")
    relative = f"{declaration['cell']}/dataset_{declaration['repetition']:04d}.tar.gz"
    archive = root / relative
    archive.parent.mkdir(exist_ok=True)
    packed = compact_directory(base, archive)
    shutil.rmtree(base)
    return {**record, **packed, "archive": relative, "manifest": relative + ".manifest.json"}


def run_study(config, *, output_dir, workers=1, preflight=None):
    config = AICDiagnosticConfig.from_dict(config.to_dict())
    root = Path(output_dir).resolve()
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four one-thread workers")
    plan = config.plan()
    if sum(r["planned_fits"] for r in plan) > 100 and not preflight:
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
