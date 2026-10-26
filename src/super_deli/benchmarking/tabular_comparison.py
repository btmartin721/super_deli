"""Frozen orchestration validation; synthetic truth never enters model selection."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from super_deli.benchmarking.density_study import CASES, generate_raw, write_raw
from super_deli.io.config import read_json, write_json
from super_deli.models.hdbscan_config import HDBSCANConfig
from super_deli.models.tabular_analysis_config import TabularAnalysisArm, TabularAnalysisConfig
from super_deli.models.tabular_cluster_config import TabularClusterConfig

VALIDATION = {
    "schema_version": 1,
    "data_seed": 79103,
    "main_seed": 79133,
    "preflight_seed": 79139,
    "model_seed": 79147,
    "cases": list(CASES),
    "sample_sizes": [64, 256],
    "datasets_per_cell": 3,
    "resampling_datasets_per_cell": 1,
    "draws": 10,
    "main_pipeline_runs": 672,
    "preflight_pipeline_runs": 88,
    "artifact_cap_bytes": 536870912,
    "min_free_bytes": 8589934592,
    "storage_allowance": 1.25,
    "max_workers": 4,
}


def model_arms(draws=0):
    names = ("trait_1", "trait_2")
    return TabularAnalysisConfig(
        (
            TabularAnalysisArm(
                "gmm",
                TabularClusterConfig(
                    True,
                    feature_names=names,
                    random_seed=VALIDATION["model_seed"],
                    n_bootstrap=draws,
                ),
            ),
            *(
                TabularAnalysisArm(
                    name,
                    HDBSCANConfig(
                        True,
                        feature_names=names,
                        random_seed=VALIDATION["model_seed"],
                        n_subsamples=draws,
                        min_samples=ms,
                    ),
                )
                for name, ms in (("hdbscan", 5), ("hdbscan_ms2", 2), ("hdbscan_ms10", 10))
            ),
        )
    )


def tasks(phase):
    if phase not in {"preflight", "main"}:
        raise ValueError("Unknown validation phase")
    root = VALIDATION[f"{phase}_seed"]
    cases = ("background",) if phase == "preflight" else CASES
    return [
        {
            "key": f"{case}_n{n}_{i:02d}",
            "phase": phase,
            "case": case,
            "n": n,
            "dataset": i,
            "data_seed": [VALIDATION["data_seed"], root, CASES.index(case), n, i],
            "draws": VALIDATION["draws"] if phase == "preflight" or i == 0 else 0,
            "replay": phase == "main" and i == 0,
        }
        for case in cases
        for n in VALIDATION["sample_sizes"]
        for i in range(1 if phase == "preflight" else VALIDATION["datasets_per_cell"])
    ]


def _replay(arm, dataset, source, directory, original):
    import numpy as np

    from super_deli.workflows.tabular_analysis import _native

    model = replace(
        arm.model, **({"n_bootstrap": 0} if arm.family == "gmm" else {"n_subsamples": 0})
    )
    _native(replace(arm, model=model), dataset, source, directory, "traits")
    assert (directory / "assignments.csv").read_bytes() == (
        original / "assignments.csv"
    ).read_bytes()
    files = ("mixture.npz", "preprocessing.npz") if arm.family == "gmm" else ("point.npz",)
    for name in files:
        with np.load(directory / name) as a, np.load(original / name) as b:
            assert a.files == b.files
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])


def _observation(task):
    import numpy as np

    from super_deli.workflows.tabular_analysis import run_tabular_analysis

    directory = Path(task["output"])
    directory.mkdir()
    result = {"task": task, "status": "failed", "error": None}
    try:
        x, truth = generate_raw(task["case"], task["n"], task["data_seed"])
        dataset = write_raw(x, directory / "raw")
        if truth is not None:
            with (directory / "truth.npz").open("xb") as handle:
                np.savez_compressed(handle, labels=truth)
        config = model_arms(task["draws"])
        summary = run_tabular_analysis(
            dataset,
            analysis_config=config,
            base_dir=directory / "raw",
            output_dir=directory / "analysis",
        )
        replays = []
        if task["replay"]:
            (directory / "replays").mkdir()
            for arm in config.arms:
                _replay(
                    arm,
                    dataset,
                    directory / "raw",
                    directory / "replays" / arm.name,
                    directory / "analysis/arms" / arm.name,
                )
                replays.append(arm.name)
        result.update(status=summary["status"], replays=replays)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    write_json(result, directory / "validation.json")
    return result


def run_validation(phase, *, output_dir, workers=1, gate_file=None):
    import shutil

    if type(workers) is not int or not 1 <= workers <= VALIDATION["max_workers"]:
        raise ValueError("Use one to four single-thread workers")
    out = Path(output_dir)
    if phase == "main":
        from super_deli.core.provenance import fingerprint_file

        gate = read_json(gate_file)
        if gate["status"] != "passed" or gate["settings"] != VALIDATION:
            raise ValueError("A matching independently audited storage preflight is required")
        if fingerprint_file(gate["preflight_audit"]).sha256 != gate["preflight_audit_sha256"]:
            raise ValueError("Preflight audit changed after storage approval")
        if shutil.disk_usage(out.parent).free < VALIDATION["min_free_bytes"]:
            raise ValueError("Insufficient free storage")
    out.mkdir()
    rows = [{**task, "output": str(out / task["key"])} for task in tasks(phase)]
    write_json(
        {
            "phase": phase,
            "settings": VALIDATION,
            "tasks": rows,
            "model_arms": model_arms(VALIDATION["draws"]).to_dict(),
        },
        out / "plan.json",
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_observation, rows))
    summary = {
        "phase": phase,
        "observations": len(results),
        "complete": sum(r["status"] == "complete" for r in results),
        "results": results,
    }
    write_json(summary, out / "validation-summary.json")
    return summary


def storage_gate(preflight_dir, audit_file, artifact_root, output_file, *, artifact_cap_bytes=None):
    import shutil

    from super_deli.core.provenance import fingerprint_file

    cap = VALIDATION["artifact_cap_bytes"] if artifact_cap_bytes is None else artifact_cap_bytes
    if type(cap) is not int or cap < 1:
        raise ValueError("artifact_cap_bytes must be a positive integer")
    probe, root = Path(preflight_dir), Path(artifact_root)
    audit = read_json(audit_file)
    if audit["status"] != "passed" or audit["phase"] != "preflight" or audit["pipeline_runs"] != 88:
        raise ValueError("Complete audited 88-pipeline preflight required")
    if read_json(probe / "plan.json")["settings"] != VALIDATION:
        raise ValueError("Preflight settings mismatch")
    # Worst observed bytes/pipeline, including the full immutable report/ingestion
    # overhead; main jobs have fewer resamples and therefore more overhead/run.
    # Use the maximum observed complete job bytes also for each point-only job.
    sizes = [
        sum(p.stat().st_size for p in (probe / t["key"]).rglob("*") if p.is_file())
        for t in tasks("preflight")
    ]
    retained = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    projection = int(
        (retained + max(sizes) * len(tasks("main")) * 2) * VALIDATION["storage_allowance"]
    )
    free = shutil.disk_usage(root).free
    passed = projection <= cap and free >= max(VALIDATION["min_free_bytes"], projection)
    result = {
        "status": "passed" if passed else "failed",
        "settings": VALIDATION,
        "preflight_audit": str(Path(audit_file).absolute()),
        "preflight_audit_sha256": fingerprint_file(audit_file).sha256,
        "artifact_cap_bytes": cap,
        "budget_amendment": cap != VALIDATION["artifact_cap_bytes"],
        "measured_job_bytes": sizes,
        "projected_total_bytes": projection,
        "retained_bytes": retained,
        "free_bytes": free,
        "method": (
            "Twice worst preflight job per main dataset (includes replay allowance), plus 25%"
        ),
    }
    write_json(result, output_file)
    return result
