"""Frozen grouped AIC/likelihood study with losslessly compact local evidence."""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

from super_deli.benchmarking.grouped_study import simulate_grouped
from super_deli.benchmarking.local_archives import compact_directory, retained_bytes, write_progress
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.io.config import write_json
from super_deli.models.grouped_power_config import GroupedPowerStudyConfig

SCOPE = (
    "Prespecified grouped AIC/likelihood power and recovery under constructed laws. "
    "Gaussian departure, selected component count, recovery and stability are distinct; "
    "none establishes species boundaries. Defaults and historical p-values unchanged."
)
COUNT_KEYS = (
    "planned_fits",
    "attempted_fits",
    "successful_fits",
    "failed_fits",
    "unattempted_fits",
)


def simulate(cell, seed, *, target_level="locality"):
    """Keep the old generator intact; vary only declared alternative mean shifts."""
    import numpy as np

    if cell.law == "cline":
        from super_deli.benchmarking.grouped_pca_study import simulate as stress

        return stress("cline", cell.n_units, seed, target_level=target_level)
    values, layout, labels = simulate_grouped(
        cell.law, cell.n_units, seed, target_level=target_level
    )
    if cell.has_groups and cell.signal != 3:
        loading = np.array([1, 0.7, 0.5, 0.3])
        targets = np.repeat(np.arange(cell.n_units), layout["counts"])
        first, second = 2 * (labels // 2) - 1, 2 * (labels % 2) - 1
        values["morphological"] += (cell.signal - 3) * first[targets, None] * loading
        values["ecological"] += (cell.signal - 3) * second[:, None] * loading
    return values, layout, labels


def policy_for(config, declaration):
    return replace(config.calibration, random_seed=declaration["null_stream_seed"])


def _verify_replay(calibration, replay, model):
    """The additional stability run must reproduce the original unsupervised point fit."""
    import numpy as np

    from super_deli.io.config import read_json

    with ZipFile(calibration / "pipeline_artifacts.zip") as archive:
        original = json.loads(archive.read("observed/run/summary.json"))
        current = read_json(replay / "summary.json")
        for field in ("unit_ids", "selected_k", "point_seed", "n_pcs", "covariance_type"):
            if original[field] != current[field]:
                raise ValueError(f"Stability point replay differs: {field}")
        if json.loads(archive.read("observed/run/candidates.json")) != read_json(
            replay / "candidates.json"
        ):
            raise ValueError("Stability candidate replay differs")
        for name in ("mixture.npz", "gmm_candidates.npz", "fusion.npz", "joint_pca.npz"):
            if "observed/run/" + name not in archive.namelist():
                continue
            with (
                np.load(io.BytesIO(archive.read("observed/run/" + name)), allow_pickle=False) as a,
                np.load(replay / name, allow_pickle=False) as b,
            ):
                if set(a.files) != set(b.files):
                    raise ValueError("Stability point parameter fields differ")
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
        if current["bootstrap"]["requested"] != model.n_bootstrap:
            raise ValueError("Stability request differs")


def _stability(base, dataset, config):
    from super_deli.io.config import read_json
    from super_deli.workflows.integrative import run_integrative

    model = replace(config.calibration.model, n_bootstrap=config.stability_draws)
    record = {
        "planned_fits": 1 + model.n_bootstrap,
        "attempted_fits": 0,
        "successful_fits": 0,
        "failed_fits": 0,
        "unattempted_fits": 1 + model.n_bootstrap,
    }
    write_json(model.to_dict(), base / "stability_model.json")
    try:
        result = run_integrative(
            dataset, model_config=model, base_dir=base / "observed", output_dir=base / "stability"
        )
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        record.update(
            status="point_failed",
            error=str(exc),
            attempted_fits=1,
            failed_fits=1,
            unattempted_fits=model.n_bootstrap,
        )
    else:
        _verify_replay(base / "calibration", base / "stability", model)
        replicas = read_json(base / "stability/bootstrap_replicates.json")
        successes = sum(r["status"] == "complete" for r in replicas)
        record.update(
            status=result["bootstrap"]["status"],
            point_replay_verified=True,
            attempted_fits=1 + len(replicas),
            successful_fits=1 + successes,
            failed_fits=len(replicas) - successes,
            unattempted_fits=model.n_bootstrap - len(replicas),
            selected_k=result["selected_k"],
            bootstrap=result["bootstrap"],
        )
    write_json(record, base / "stability_record.json")
    return record


def _observation(task):
    import numpy as np
    from threadpoolctl import threadpool_limits

    from super_deli.io.config import read_json
    from super_deli.models.grouped_null import write_raw_views
    from super_deli.workflows.calibration_artifacts import fit_counts
    from super_deli.workflows.grouped_likelihood import run_grouped_likelihood

    source, raw_config, declaration = task
    root = Path(source)
    config = GroupedPowerStudyConfig.from_dict(raw_config)
    cell = next(c for c in config.cells if c.name == declaration["cell"])
    i = declaration["repetition"]
    base = root / ".working" / f"{cell.name}_{i:04d}"
    if shutil.disk_usage(root).free < config.min_free_bytes + 256 * 1024**2:
        raise RuntimeError("Insufficient working space; preserve existing evidence")
    base.mkdir(parents=True, exist_ok=False)
    write_json(declaration, base / "declaration.json")
    start = perf_counter()
    with (
        threadpool_limits(limits=1),
        (base / "stdout.log").open("x") as out,
        (base / "stderr.log").open("x") as err,
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
    ):
        values, layout, labels = simulate(
            cell, declaration["data_seed"], target_level=config.calibration.model.target_level
        )
        dataset = write_raw_views(values, layout, base / "observed")
        with (base / "values.npz").open("xb") as handle:
            np.savez(handle, **values)
        write_json(layout, base / "generating_layout.json")
        write_json(
            {
                "unit_ids": layout["unit_ids"],
                "has_groups": cell.has_groups,
                "labels": labels.tolist(),
                "scope": "Generating labels for post-fit evaluation only; not species",
            },
            base / "truth.json",
        )
        policy = policy_for(config, declaration)
        try:
            result = run_grouped_likelihood(
                dataset, config=policy, base_dir=base / "observed", output_dir=base / "calibration"
            )
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            write_json(
                {"error": str(exc), "traceback": traceback.format_exc()},
                base / "outer_failure.json",
            )
            point = (
                read_json(base / "calibration/observed.json")
                if (base / "calibration/observed.json").exists()
                else None
            )
            draws = [read_json(p) for p in sorted((base / "calibration").glob("draw_*.json"))]
            calibration = {
                "status": "failed",
                "p_value": None,
                "error": str(exc),
                **fit_counts(point, draws, policy.n_simulations),
            }
        else:
            calibration = {
                "status": result["inference"]["status"],
                "p_value": result["inference"]["p_value"],
                **{key: result[key] for key in COUNT_KEYS},
            }
        stability = None
        if declaration["stability"]:
            if (base / "calibration/observed.json").exists() and read_json(
                base / "calibration/observed.json"
            )["status"] == "complete":
                stability = _stability(base, dataset, config)
            else:
                stability = {
                    "status": "not_attempted",
                    "planned_fits": 1 + config.stability_draws,
                    "attempted_fits": 0,
                    "successful_fits": 0,
                    "failed_fits": 0,
                    "unattempted_fits": 1 + config.stability_draws,
                }
                write_json(stability, base / "stability_record.json")
        record = {
            **declaration,
            "cell_config": cell.to_dict(),
            "calibration": calibration,
            "stability_record": stability,
            "elapsed_seconds": perf_counter() - start,
            "original_directory": str(base),
            "values_sha256": fingerprint_file(base / "values.npz").sha256,
        }
        record.update(
            {key: calibration[key] + (stability[key] if stability else 0) for key in COUNT_KEYS}
        )
        if record["planned_fits"] != declaration["planned_fits"]:
            raise ValueError("Observation fit accounting differs from frozen plan")
        write_json(record, base / "record.json")
    destination = root / cell.name / f"dataset_{i:04d}.tar.gz"
    packed = compact_directory(base, destination)
    # Only this new study's verified redundant working copy is removed.
    shutil.rmtree(base)
    return {
        **record,
        **packed,
        "archive": str(destination.relative_to(root)),
        "manifest": str(
            destination.with_name(destination.name + ".manifest.json").relative_to(root)
        ),
    }


def run_study(config, *, output_dir, workers=1, preflight=None):
    config = GroupedPowerStudyConfig.from_dict(config.to_dict())
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four independent one-thread jobs")
    if sum(r["planned_fits"] for r in config.plan()) > 1000 and preflight is None:
        raise ValueError("A matching storage preflight is required before a larger study")
    root = Path(output_dir).resolve()
    if preflight is not None and (
        preflight["status"] != "passed"
        or preflight["study_sha256"] != configuration_hash(config.to_dict())
    ):
        raise ValueError("Storage preflight does not match the frozen study declaration")
    artifact_root = Path(preflight["artifact_root"]).resolve() if preflight else root
    if preflight and (root == artifact_root or not root.is_relative_to(artifact_root)):
        raise ValueError("Full study output must be inside its checked artifact root")
    root.mkdir(parents=True, exist_ok=False)
    write_json(config.to_dict(), root / "study.json")
    plan = config.plan()
    write_json(plan, root / "plan.json")
    records, pending = [], {}
    tasks = iter(plan)
    start = perf_counter()

    def progress(status):
        result = {
            "schema_version": 1,
            "scope": SCOPE,
            "status": status,
            "observations": len(records),
            "requested_observations": len(plan),
            "planned_fits": sum(r["planned_fits"] for r in plan),
            **{key: sum(r[key] for r in records) for key in COUNT_KEYS if key != "planned_fits"},
            "retained_archive_bytes": sum(r["size_bytes"] for r in records),
            "elapsed_seconds": perf_counter() - start,
        }
        write_progress(result, root / "progress.json")
        return result

    with (root / "observations.jsonl").open("x", encoding="utf-8") as ledger:
        with ProcessPoolExecutor(max_workers=workers) as executor:

            def submit():
                declaration = next(tasks, None)
                if declaration is not None:
                    future = executor.submit(
                        _observation, (str(root), config.to_dict(), declaration)
                    )
                    pending[future] = declaration
                    return True
                return False

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
                            raise RuntimeError(
                                "Artifact cap reached; preserve partial study without replacements"
                            )
                        submit()
            except BaseException as exc:
                for future in pending:
                    future.cancel()
                write_json(
                    {
                        **progress("interrupted"),
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "pending": list(pending.values()),
                    },
                    root / "execution_failure.json",
                )
                raise
    result = progress("complete")
    write_json(result, root / "summary.json")
    return result
