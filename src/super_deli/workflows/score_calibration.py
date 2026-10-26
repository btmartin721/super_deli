"""Immutable score-calibration artifacts and a read-only integrative-run adapter."""

from __future__ import annotations

import csv
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.results import Evidence
from super_deli.io.config import read_json, write_json
from super_deli.models.calibration_config import GaussianCalibrationConfig
from super_deli.models.integrative_config import IntegrativeConfig, require_calibrated_pca


def write_score_calibration(fit, directory, *, provenance):
    import numpy as np

    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    with (directory / "null_model.npz").open("xb") as handle:
        np.savez(handle, mean=fit.mean, covariance=fit.covariance, scores=fit.scores)
    write_json(fit.records, directory / "null_replicates.json")
    inference = fit.summary["inference"]
    evidence = Evidence(
        "p_value",
        "single_gaussian_frozen_scores",
        "available" if inference["p_value"] is not None else "unavailable",
        method=inference["p_value_kind"],
        value=inference["p_value"],
        replicates=inference["requested"],
        null_hypothesis=fit.summary["null_hypothesis"],
        multiplicity=(
            "none across analyses; one global statistic incorporates the configured K search"
        ),
        reason="A requested null refit failed; see failed-outcome bounds"
        if inference["failed"]
        else None,
    )
    write_json(asdict(evidence), directory / "evidence.json")
    summary = {
        **fit.summary,
        "provenance": provenance,
        "software": {
            name: version(name)
            for name in ("super-deli", "numpy", "scipy", "scikit-learn", "threadpoolctl")
        },
        "artifacts": {
            name: fingerprint_file(directory / name).sha256
            for name in ("null_model.npz", "null_replicates.json", "evidence.json")
        },
    }
    write_json(summary, directory / "summary.json")
    return summary


def _source_scores(source, config):
    import numpy as np

    summary = read_json(source / "summary.json")
    if summary.get("covariance_type") == "tied":
        raise ValueError("Shared-covariance significance calibration is not yet supported")
    if (
        summary.get("schema_version") != 1
        or summary.get("status") != "complete"
        or summary.get("model") != "integrative_weighted_pca_full_gmm"
        or summary.get("covariance_type") != "full"
    ):
        raise ValueError("A completed, supported integrative clustering run is required")
    provenance = summary["provenance"]
    if configuration_hash(provenance["configuration"]) != provenance["configuration_sha256"]:
        raise ValueError("Source configuration checksum mismatch")
    search = IntegrativeConfig.from_dict(provenance["configuration"]["model"])
    require_calibrated_pca(search)
    for name in ("mixture", "fusion"):
        if fingerprint_file(source / f"{name}.npz").sha256 != provenance[f"{name}_sha256"]:
            raise ValueError(f"Source {name} checksum mismatch")
    with np.load(source / "mixture.npz", allow_pickle=False) as arrays:
        scores = arrays["scores"]
    n, d = len(summary["unit_ids"]), search.n_pcs
    if len(set(summary["unit_ids"])) != n or scores.shape != (n, d):
        raise ValueError("Source score dimensions or unit identities disagree")
    if summary["n_pcs"] != d or 8 * 24 * n * d > config.max_working_bytes:
        raise ValueError("Source score dimension/workspace exceeds the declared contract")
    with np.load(source / "fusion.npz", allow_pickle=False) as arrays:
        projected = (arrays["fused"] - arrays["mean"]) @ arrays["components"].T
    if not np.allclose(scores, projected, rtol=1e-10, atol=1e-12):
        raise ValueError("Saved scores disagree with the frozen fusion/PCA parameters")
    with (source / "scores.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if (
        rows[0] != ["unit_id", *(f"PC{i + 1}" for i in range(d))]
        or [row[0] for row in rows[1:]] != summary["unit_ids"]
        or not np.array_equal(scores, np.array([row[1:] for row in rows[1:]], dtype=float))
    ):
        raise ValueError("Source score CSV differs in unit order or values")
    return scores, search, summary


def run_integrative_calibration(reference_dir, *, config, output_dir):
    config = GaussianCalibrationConfig.from_dict(config.to_dict())
    source, destination = Path(reference_dir).resolve(), Path(output_dir).resolve()
    if destination == source or destination.is_relative_to(source):
        raise ValueError("Calibration output must be outside the immutable source run")
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    try:
        from super_deli.models.gaussian_calibration import (
            bic_improvement,
            calibrate_gaussian_scores,
        )
    except (ImportError, OSError) as exc:
        raise RuntimeError("Install 'super-deli[clustering]' for score calibration") from exc
    if read_json(source / "summary.json").get("model") == "tabular_pca_hdbscan":
        raise ValueError("HDBSCAN significance calibration is not supported")
    files = ("summary.json", "mixture.npz", "fusion.npz", "scores.csv", "candidates.json")
    before = {name: fingerprint_file(source / name).sha256 for name in files}
    try:
        scores, search, original = _source_scores(source, config)
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Malformed source integrative artifacts: {exc}") from exc
    fit = calibrate_gaussian_scores(
        scores,
        search=search,
        config=config,
        point_seed=original["point_seed"],
        expected_point=(
            original["selected_k"],
            bic_improvement(read_json(source / "candidates.json")),
        ),
    )
    if before != {name: fingerprint_file(source / name).sha256 for name in files}:
        raise ValueError("Source run changed during calibration")
    return write_score_calibration(
        fit,
        destination,
        provenance={
            "reference_run": str(source),
            "source_sha256": before,
            "target_level": original["level"],
            "unit_ids": original["unit_ids"],
            "excluded_units": original["excluded_units"],
            "conditioning": "Retained cohort and frozen scores; no raw-data pipeline refits",
        },
    )
