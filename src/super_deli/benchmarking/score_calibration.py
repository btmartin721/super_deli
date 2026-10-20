"""Independent, prespecified validation of the Gaussian score-space approximation."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.study_analysis import rate_summary
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.io.config import write_json
from super_deli.models.calibration_config import GaussianCalibrationConfig
from super_deli.models.tabular_cluster_config import TabularClusterConfig
from super_deli.workflows.score_calibration import write_score_calibration

SCENARIOS = ("spherical_gaussian", "correlated_gaussian", "separated_mixture", "heavy_tailed")


def study_options(value):
    required = {
        "schema_version",
        "n_repetitions",
        "n_units",
        "random_seed",
        "calibration",
        "search",
    }
    if not isinstance(value, dict) or value.keys() != required or value["schema_version"] != 1:
        raise ValueError("Score study requires exactly its supported versioned fields")
    for name in ("n_repetitions", "n_units", "random_seed"):
        if type(value[name]) is not int or value[name] < (0 if name == "random_seed" else 1):
            raise ValueError(f"Invalid score study {name}")
    calibration = GaussianCalibrationConfig.from_dict(value["calibration"])
    search = TabularClusterConfig.from_dict(value["search"])
    if (
        search.n_pcs != 2
        or search.n_bootstrap != 0
        or search.max_k < 2
        or value["n_units"] % 4
        or value["n_units"] < max(16, 4 * search.min_cluster_size)
        or search.feature_names
        or search.missing_policy != "error"
        or search.scaling != "center"
        or search.count_transform != "identity"
    ):
        raise ValueError("Score study requires 2D complete scores and no preprocessing")
    return calibration, search


def simulate_scores(scenario, n, seed):
    import numpy as np

    if scenario not in SCENARIOS:
        raise ValueError("Unknown Gaussian score validation scenario")
    rng = np.random.default_rng(seed)
    if scenario == "spherical_gaussian":
        return rng.normal(size=(n, 2))
    factor = np.linalg.cholesky([[4.0, 1.6], [1.6, 1.0]])
    if scenario == "correlated_gaussian":
        return np.array([2.0, -1.0]) + rng.normal(size=(n, 2)) @ factor.T
    if scenario == "separated_mixture":
        centers = np.array([[-4.0, -4.0], [-4.0, 4.0], [4.0, -4.0], [4.0, 4.0]])
        return centers[rng.integers(4, size=n)] + rng.normal(scale=0.35, size=(n, 2))
    # Elliptical, unimodal multivariate Student t_3: Gaussian-null misspecification.
    return (rng.normal(size=(n, 2)) @ factor.T) / np.sqrt(rng.chisquare(3, size=(n, 1)) / 3)


def run_score_study(value, *, output_dir, scenario):
    try:
        from super_deli.models.gaussian_calibration import calibrate_gaussian_scores, score_hash
    except (ImportError, OSError) as exc:
        raise RuntimeError("Install 'super-deli[clustering]' for the score study") from exc

    calibration, search = study_options(value)
    if scenario not in SCENARIOS:
        raise ValueError("Select one declared score-study scenario")
    directory = Path(output_dir).resolve()
    directory.mkdir(exist_ok=False)
    write_json(value, directory / "study.json")
    rows = []
    for repetition in range(value["n_repetitions"]):
        data_seed = derived_seed(value["random_seed"], scenario, repetition, "data")
        fit_seed = derived_seed(value["random_seed"], scenario, repetition, "calibration")
        scores = simulate_scores(scenario, value["n_units"], data_seed)
        row = {
            "scenario": scenario,
            "repetition": repetition,
            "data_seed": data_seed,
            "calibration_seed": fit_seed,
            "scores_sha256": score_hash(scores),
            "directory": f"rep_{repetition:04d}",
        }
        try:
            fit = calibrate_gaussian_scores(
                scores, search=search, config=replace(calibration, random_seed=fit_seed)
            )
            result = write_score_calibration(fit, directory / row["directory"], provenance=row)
            inference = result["inference"]
            row.update(
                status=inference["status"],
                p_value=inference["p_value"],
                reject=inference["reject_single_gaussian"],
                statistic=result["observed_statistic"],
                selected_k=result["observed_selected_k"],
                successful_draws=inference["successful"],
                failed_draws=inference["failed"],
                summary_sha256=fingerprint_file(
                    directory / row["directory"] / "summary.json"
                ).sha256,
            )
        except (ValueError, FloatingPointError) as exc:
            row.update(status="point_failed", error=str(exc), successful_draws=0, failed_draws=0)
        rows.append(row)
        write_json(rows, directory / f"ledger_{repetition:04d}.json")
    summary = {
        "schema_version": 1,
        "status": "complete",
        "scenario": scenario,
        "configuration_sha256": configuration_hash(value),
        "requested": value["n_repetitions"],
        "datasets": rows,
        "rejection_rate": rate_summary(
            [int(r["reject"]) if r.get("reject") is not None else None for r in rows], len(rows)
        ),
        "rate_interpretation": (
            "Type-I error under the specified Gaussian generator"
            if scenario.endswith("gaussian")
            else "Power for the specified mixture"
            if scenario == "separated_mixture"
            else "Gaussian misspecification sensitivity; NOT type-I error for species/clustering"
        ),
        "attempted_null_draws": sum(r["successful_draws"] + r["failed_draws"] for r in rows),
        "successful_null_draws": sum(r["successful_draws"] for r in rows),
        "planned_null_draws": len(rows) * calibration.n_simulations,
    }
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (directory / "datasets.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(summary, directory / "summary.json")
    return summary
