"""Independent observed datasets and fresh reference draws, without shared null banks."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.study_analysis import rate_summary
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.io.config import read_json, write_json
from super_deli.models.pipeline_config import CASES, PipelineStudyConfig


def case_config(study, case, repetition):
    if case not in CASES or not 0 <= repetition < study.n_datasets:
        raise ValueError("Unknown validation case or repetition")
    policy = study.calibration.for_linkage(case.startswith("grouped"))
    return replace(
        policy, random_seed=derived_seed(study.random_seed, case, repetition, "null_stream")
    )


def summarize_records(records, study, case, *, interpretation=None):
    available = [r for r in records if r["p_value"] is not None]
    return {
        "schema_version": 1,
        "status": "complete",
        "case": case,
        "configuration_sha256": configuration_hash(study.to_dict()),
        "requested_datasets": study.n_datasets,
        "available_tests": len(available),
        "datasets": records,
        "planned_fits": study.n_datasets * (study.calibration.n_simulations + 1),
        **{
            key: sum(r[key] for r in records)
            for key in ("attempted_fits", "successful_fits", "failed_fits", "unattempted_fits")
        },
        "rejection_rate": rate_summary(
            [int(r["reject"]) if r["reject"] is not None else None for r in records],
            study.n_datasets,
        ),
        "interpretation": interpretation
        or (
            "Reference-law type-I error"
            if case.endswith("_null")
            else "Power for complementary four-group alternative"
        ),
    }


def run_pipeline_study(study, *, case, output_dir):
    study = PipelineStudyConfig.from_dict(study.to_dict())
    if case not in CASES:
        raise ValueError("Unknown validation case")
    from super_deli.workflows.pipeline_calibration import _dependencies, run_reference_calibration

    simulate, write_simulation, *_ = _dependencies()
    root = Path(output_dir).resolve()
    root.mkdir(exist_ok=False)
    write_json(study.to_dict(), root / "study.json")
    records = []
    with (root / "datasets.jsonl").open("x", encoding="utf-8") as ledger:
        for i in range(study.n_datasets):
            policy = case_config(study, case, i)
            seed = derived_seed(study.random_seed, case, i, "observed_data")
            directory = root / f"dataset_{i:04d}"
            directory.mkdir()
            data = simulate(
                policy.reference.scenario("null" if case.endswith("_null") else "complementary"),
                policy.reference,
                seed,
            )
            write_simulation(data, directory / "observed_simulation")
            dataset = config_from_dict(
                read_json(directory / "observed_simulation/observed/dataset.json")
            )
            result = run_reference_calibration(
                dataset,
                config=policy,
                base_dir=directory / "observed_simulation/observed",
                output_dir=directory / "calibration",
            )
            record = {
                "case": case,
                "repetition": i,
                "data_seed": seed,
                "null_stream_seed": policy.random_seed,
                "directory": directory.name,
                "status": result["inference"]["status"],
                "p_value": result["inference"]["p_value"],
                "reject": result["inference"]["reject_reference_population"],
                "statistic": result["point"].get("statistic"),
                "selected_k": result["point"].get("selected_k"),
                **{
                    key: result[key]
                    for key in (
                        "planned_fits",
                        "attempted_fits",
                        "successful_fits",
                        "failed_fits",
                        "unattempted_fits",
                    )
                },
                "summary_sha256": fingerprint_file(directory / "calibration/summary.json").sha256,
            }
            records.append(record)
            ledger.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            ledger.flush()
    summary = summarize_records(records, study, case)
    write_json(summary, root / "summary.json")
    return summary
