"""Immutable density-specific artifacts; GMM evidence contracts are not reused."""

from __future__ import annotations

import csv
from importlib.metadata import version
from pathlib import Path

from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.io.config import write_json
from super_deli.io.tabular import ingest_tabular
from super_deli.models.hdbscan_config import HDBSCANConfig


def run_tabular_hdbscan(dataset, *, model_config, base_dir, output_dir, modality_name=None):
    dataset = config_from_dict(dataset.to_dict())
    config = HDBSCANConfig.from_dict(model_config.to_dict())
    selected = [
        m
        for m in dataset.modalities
        if m.kind in {"morphological", "ecological"}
        and (modality_name is None or m.name == modality_name)
    ]
    if len(selected) != 1:
        raise ValueError("Select exactly one morphology/ecology table with --modality")
    try:
        import numpy as np

        from super_deli.models.tabular_hdbscan import fit_tabular_hdbscan, point_arrays
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Density clustering requires: pip install 'super-deli[clustering]'"
        ) from exc
    out = Path(output_dir).absolute()
    out.mkdir(exist_ok=False)
    try:
        loaded = ingest_tabular(
            dataset,
            base_dir=base_dir,
            output_dir=out / "ingestion",
            modality_names=[selected[0].name],
        )
        table = loaded.tables[0]
        fit = fit_tabular_hdbscan(table, config=config)
        with (out / "point.npz").open("xb") as f:
            np.savez_compressed(f, **point_arrays(fit.point))
        with (out / "stability.npz").open("xb") as f:
            np.savez_compressed(f, **fit.stability.arrays)
        write_json(fit.stability.records, out / "subsamples.json")
        write_json(
            {
                "feature_order": fit.point.model.feature_names,
                "feature_types": fit.point.model.feature_types,
                "excluded_features": fit.point.model.exclusions,
                "nonselected_columns": [
                    f.name for f in table.features if f.name not in fit.point.model.feature_names
                ],
            },
            out / "preprocessing.json",
        )
        with (out / "assignments.csv").open("x", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(
                [
                    "observation_id",
                    "unit_id",
                    "cluster",
                    "density_membership_strength",
                    *(f"PC{i + 1}" for i in range(config.n_pcs)),
                ]
            )
            for obs, unit, label, strength, scores in zip(
                fit.observation_ids,
                fit.unit_ids,
                fit.point.labels,
                fit.point.strength,
                fit.point.scores,
                strict=True,
            ):
                writer.writerow([obs, unit, int(label), float(strength), *scores])
        configuration = {
            "dataset": dataset.to_dict(),
            "model": config.to_dict(),
            "modality": table.modality_name,
        }
        summary = {
            "schema_version": 1,
            "status": "complete",
            "model": config.model_family,
            "selected_k": fit.point.selected_k,
            "n_pcs": config.n_pcs,
            "level": table.level,
            "unit_ids": fit.unit_ids,
            "observation_ids": fit.observation_ids,
            "noise_count": int(np.sum(fit.point.labels < 0)),
            "noise_label": -1,
            "stability": fit.stability.summary,
            "inference": {
                "status": "not_available",
                "p_value": None,
                "reason": "No significance calibration for this density family",
            },
            "limitations": [
                "Candidate density groups, noise and strength do not establish species boundaries",
                "Strength measures algorithmic density membership, not species probability",
                "Stability is conditional on the input cohort, subsampling and frozen settings",
                "No predictions for absent or new units; noise is never a shared cluster",
                "Independence is declared by the analyst and is not verified by the model",
                "Scaling, retained PCs and density parameters affect groups and noise",
            ],
            "provenance": {
                "configuration": configuration,
                "configuration_sha256": configuration_hash(configuration),
                "ingestion": loaded.summary["provenance"],
                "software": {
                    n: version(n)
                    for n in ("super-deli", "numpy", "scipy", "scikit-learn", "threadpoolctl")
                },
            },
            "artifacts": {p.name: fingerprint_file(p).sha256 for p in out.iterdir() if p.is_file()},
        }
        write_json(summary, out / "summary.json")
        return summary
    except Exception as exc:
        write_json({"status": "failed", "error": str(exc)}, out / "failure.json")
        raise
