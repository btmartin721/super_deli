"""Execute declared studies without changing models or selecting settings using truth."""

from __future__ import annotations

import csv
import importlib.util
import json
import platform
from dataclasses import replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from super_deli.benchmarking.config import BenchmarkConfig, derived_seed
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.io.config import write_json


def run_benchmark(config: BenchmarkConfig, *, output_dir: str | Path):
    config = BenchmarkConfig.from_dict(config.to_dict())
    try:
        import numpy as np

        from super_deli.benchmarking.metrics import score_run, summarize_runs
        from super_deli.benchmarking.simulation import simulate, write_simulation
        from super_deli.workflows.integrative import run_integrative
    except (ImportError, OSError) as exc:
        raise RuntimeError("Install 'super-deli[clustering]' for integrative benchmarking") from exc
    if (
        any("genomic" in arm.kinds for arm in config.arms)
        and importlib.util.find_spec("snpio") is None
    ):
        raise RuntimeError("Genomic benchmark arms require 'super-deli[genomics]'")
    destination = Path(output_dir).absolute()
    destination.mkdir(exist_ok=False)
    configuration = config.to_dict()
    write_json(configuration, destination / "config.json")
    records = []
    try:
        with (destination / "runs.jsonl").open("x", encoding="utf-8") as log:
            for scenario in config.scenarios:
                for repetition in range(config.n_datasets):
                    simulation_seed = derived_seed(
                        config.random_seed, scenario.name, repetition, "simulation"
                    )
                    analysis_seed = derived_seed(
                        config.random_seed, scenario.name, repetition, "analysis"
                    )
                    directory = destination / f"{scenario.name}_{repetition:04d}"
                    directory.mkdir()
                    data = simulate(scenario, config, simulation_seed)
                    dataset = write_simulation(data, directory / "simulation")
                    truth = {
                        t: {
                            "primary": data.primary_labels[i],
                            "alternative": data.alternative_labels[i]
                            if data.alternative_labels
                            else None,
                        }
                        for i, t in enumerate(data.target_ids)
                    }
                    for arm in config.arms:
                        grouped = scenario.linkage == "grouped"
                        model = replace(
                            config.model,
                            target_level="locality" if grouped else "specimen",
                            missing_views="error",
                            random_seed=analysis_seed,
                            views=tuple(
                                replace(
                                    v,
                                    aggregation="mean"
                                    if grouped and v.kind != "ecological"
                                    else "none",
                                )
                                for v in config.model.views
                                if v.kind in arm.kinds
                            ),
                        )
                        output = directory / arm.name
                        record = {
                            "scenario": scenario.name,
                            "arm": arm.name,
                            "repetition": repetition,
                            "simulation_seed": simulation_seed,
                            "analysis_seed": analysis_seed,
                            "run_path": str(output.relative_to(destination)),
                            "simulation_manifest_sha256": fingerprint_file(
                                directory / "simulation/manifest.json"
                            ).sha256,
                        }
                        started = perf_counter()
                        try:
                            result = run_integrative(
                                config_from_dict(dataset),
                                model_config=model,
                                base_dir=directory / "simulation/cohort",
                                output_dir=output,
                            )
                        except (
                            ValueError,
                            RuntimeError,
                            np.linalg.LinAlgError,
                            FloatingPointError,
                        ) as exc:
                            record.update(
                                status="failed",
                                error=str(exc),
                                error_type=type(exc).__name__,
                                error_stage="integrative_workflow",
                            )
                        else:
                            # This is a study invariant, not a numerical model failure.
                            if result["unit_ids"] != list(data.common_ids):
                                raise ValueError(
                                    "All arms must fit the same ordered complete-case cohort"
                                )
                            record.update(status="complete", evaluation=score_run(output, truth))
                            write_json(record["evaluation"], output / "evaluation.json")
                        record["elapsed_seconds"] = perf_counter() - started
                        records.append(record)
                        log.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                        log.flush()
        attempted = len(config.scenarios) * config.n_datasets * len(config.arms)
        if len(records) != attempted:
            raise ValueError("Incomplete benchmark run ledger")
        summary = {
            "schema_version": 1,
            "status": "complete",
            "attempted": attempted,
            "point_failed": sum(r["status"] == "failed" for r in records),
            "failure_records": [
                {
                    k: r[k]
                    for k in ("scenario", "arm", "repetition", "error", "error_type", "error_stage")
                }
                for r in records
                if r["status"] == "failed"
            ],
            "bootstrap_incomplete_runs": sum(
                r["status"] == "complete" and r["evaluation"]["bootstrap"]["status"] != "available"
                for r in records
            ),
            "created_at": datetime.now(UTC).isoformat(),
            "configuration_sha256": configuration_hash(configuration),
            "runs_sha256": fingerprint_file(destination / "runs.jsonl").sha256,
            "software": {
                "python": platform.python_version(),
                **{
                    name: version(name) for name in ("super-deli", "numpy", "scikit-learn", "scipy")
                },
                **({"snpio": version("snpio")} if importlib.util.find_spec("snpio") else {}),
            },
            "interpretation": [
                "Completed study execution does not imply model success or biological validity",
                "Generating partitions are constructed targets, not empirical species truth",
                "Equal dataset weighting; MCSE uses independent datasets, not pairs or inner draws",
                "Means condition on availability; missing/failure denominators stay explicit",
                "Paired comparisons use the same cohort, dataset and target bootstrap streams",
                "Coassignment error/support bins describe recovery, not species calibration",
                "OOB errors condition on observed unaffected pairs; inspect pair coverage",
                "Bootstrap uncertainty holds source members and genomic locus eligibility fixed",
                "Small studies and zero estimated MCSE do not establish exact performance",
            ],
            **summarize_runs(records, config),
        }
        with (destination / "metrics.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                (
                    "scenario",
                    "arm",
                    "metric",
                    "attempted",
                    "n_available",
                    "n_missing",
                    "mean",
                    "mcse",
                    "sd",
                    "median",
                )
            )
            for group in summary["groups"]:
                for metric, values in group["metrics"].items():
                    writer.writerow(
                        (
                            group["scenario"],
                            group["arm"],
                            metric,
                            group["attempted"],
                            *(
                                values[k]
                                for k in (
                                    "n_available",
                                    "n_missing",
                                    "mean",
                                    "mcse",
                                    "sd",
                                    "median",
                                )
                            ),
                        )
                    )
        write_json(summary, destination / "summary.json")
        return summary
    except Exception as exc:
        write_json({"status": "failed", "error": str(exc)}, destination / "failure.json")
        raise
