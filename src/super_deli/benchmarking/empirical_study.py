"""Independent raw-data validation of the fitted continuous-view null."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.pipeline_study import summarize_records
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json
from super_deli.models.empirical_config import CASES, EmpiricalStudyConfig


def simulate_continuous(case, n, seed):
    import numpy as np

    if case not in CASES or type(n) is not int or n < 20 or n % 4:
        raise ValueError("Use a supported case and >=20 targets divisible by four")
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 8))
    labels = np.zeros(n, dtype=int)
    if case == "spherical_gaussian":
        values = z
    elif case == "complementary":
        labels = rng.permutation(np.repeat(np.arange(4), n // 4))
        centers = np.column_stack((2 * (labels // 2) - 1, 2 * (labels % 2) - 1))
        effects = np.concatenate(
            [centers[:, j, None] * np.array([1, 0.7, 0.5, 0.3]) for j in range(2)], axis=1
        )
        values = 3 * effects + z
    else:
        correlation = 0.6 ** np.abs(np.arange(8)[:, None] - np.arange(8)[None, :])
        scale = np.linspace(0.5, 2, 8)
        values = (z @ np.linalg.cholesky(correlation).T) * scale
        if case == "heavy_tailed":
            values = values / np.sqrt(rng.chisquare(3, size=(n, 1)) / 3)
        values += np.linspace(-2, 2, 8)
    units = tuple(f"unit_{i:04d}" for i in range(n))
    layout = {
        "schema_version": 1,
        "unit_ids": units,
        "views": [
            {
                "modality": kind,
                "kind": kind,
                "feature_names": [f"feature_{j}" for j in range(4)],
                "feature_units": [None] * 4,
                "unit_ids": units,
                "observation_ids": [f"{kind}_{u}" for u in units],
            }
            for kind in ("morphological", "ecological")
        ],
    }
    return values, layout, labels


def case_policy(study, case, repetition):
    if case not in CASES or not 0 <= repetition < study.n_datasets:
        raise ValueError("Unknown empirical validation case/repetition")
    return replace(
        study.calibration,
        random_seed=derived_seed(study.random_seed, case, repetition, "null_stream"),
    )


def summarize_empirical(records, study, case):
    label = (
        "Type-I error under the specified joint raw Gaussian"
        if case.endswith("gaussian")
        else "Power for the complementary four-group alternative"
        if case == "complementary"
        else "Gaussian misspecification sensitivity; NOT species/clustering type-I error"
    )
    return summarize_records(records, study, case, interpretation=label)


def run_empirical_study(study, *, case, output_dir):
    from super_deli.workflows.empirical_calibration import _dependencies, run_empirical_calibration

    study = EmpiricalStudyConfig.from_dict(study.to_dict())
    if case not in CASES:
        raise ValueError("Unknown empirical validation case")
    np, _, _, write_raw_views, _, _ = _dependencies()
    root = Path(output_dir).resolve()
    root.mkdir(exist_ok=False)
    write_json(study.to_dict(), root / "study.json")
    records = []
    with (root / "datasets.jsonl").open("x", encoding="utf-8") as ledger:
        for i in range(study.n_datasets):
            policy = case_policy(study, case, i)
            seed = derived_seed(study.random_seed, case, i, "observed_data")
            directory = root / f"dataset_{i:04d}"
            directory.mkdir()
            values, layout, labels = simulate_continuous(case, study.n_units, seed)
            dataset = write_raw_views(values, layout, directory / "observed")
            with (directory / "values.npy").open("xb") as handle:
                np.save(handle, values, allow_pickle=False)
            write_json(
                {
                    "scope": "Constructed generating partition, not species identities",
                    "labels": labels.tolist(),
                },
                directory / "truth.json",
            )
            row = {
                "case": case,
                "repetition": i,
                "data_seed": seed,
                "null_stream_seed": policy.random_seed,
                "directory": directory.name,
                "values_sha256": fingerprint_file(directory / "values.npy").sha256,
            }
            try:
                result = run_empirical_calibration(
                    dataset,
                    config=policy,
                    base_dir=directory / "observed",
                    output_dir=directory / "calibration",
                )
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                failure = read_json(directory / "calibration/failure.json")
                row.update(
                    status="failed",
                    error=str(exc),
                    p_value=None,
                    reject=None,
                    statistic=None,
                    selected_k=None,
                    **{
                        k: failure[k]
                        for k in (
                            "planned_fits",
                            "attempted_fits",
                            "successful_fits",
                            "failed_fits",
                            "unattempted_fits",
                        )
                    },
                    outcome_file="failure.json",
                )
            else:
                row.update(
                    status=result["inference"]["status"],
                    p_value=result["inference"]["p_value"],
                    reject=result["inference"]["reject_joint_raw_gaussian"],
                    statistic=result["point"].get("statistic"),
                    selected_k=result["point"].get("selected_k"),
                    **{
                        k: result[k]
                        for k in (
                            "planned_fits",
                            "attempted_fits",
                            "successful_fits",
                            "failed_fits",
                            "unattempted_fits",
                        )
                    },
                    outcome_file="summary.json",
                )
            row["outcome_sha256"] = fingerprint_file(
                directory / "calibration" / row["outcome_file"]
            ).sha256
            records.append(row)
            ledger.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            ledger.flush()
    summary = summarize_empirical(records, study, case)
    write_json(summary, root / "summary.json")
    return summary
