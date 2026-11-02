"""Explicitly experimental spectral artifacts; withheld output is never K=1."""

import csv
from importlib.metadata import version
from pathlib import Path

from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.io.config import write_json
from super_deli.io.tabular import ingest_tabular
from super_deli.models.spectral_config import SpectralConfig


def run_tabular_spectral(dataset, *, model_config, base_dir, output_dir, modality_name=None):
    dataset = config_from_dict(dataset.to_dict())
    config = SpectralConfig.from_dict(model_config.to_dict())
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

        from super_deli.models.tabular_spectral import fit_tabular_spectral
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Spectral clustering requires: pip install 'super-deli[clustering]'"
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
        result, arrays = fit_tabular_spectral(table, config=config)
        with (out / "point.npz").open("xb") as f:
            np.savez_compressed(f, **arrays)
        if result["status"] == "available":
            with (out / "assignments.csv").open("x", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(["observation_id", "unit_id", "cluster"])
                writer.writerows(
                    zip(table.observation_ids, table.unit_ids, arrays["labels"], strict=True)
                )
        configuration = {
            "dataset": dataset.to_dict(),
            "model": config.to_dict(),
            "modality": table.modality_name,
        }
        summary = {
            "schema_version": 1,
            "model": config.model_family,
            **result,
            "level": table.level,
            "unit_ids": table.unit_ids,
            "observation_ids": table.observation_ids,
            "feature_order": config.feature_names,
            "preprocessing": "Complete raw features; standard SD ddof=0; no PCA",
            "inference": {
                "p_value": None,
                "status": "not_available",
                "reason": "Experimental point-only heuristic, no species-significance test",
            },
            "stability": {"status": "not_computed", "reason": "Point-only research workflow"},
            "limitations": [
                "Eigengap selection and K=1 do not establish species boundaries or a true null",
                "Numerical abstention is not calibrated statistical uncertainty",
                "Graph construction, measurement scale and finite search range affect the result",
                "No significance, partition confidence, resampling support or new-unit prediction",
                "Independent biological units are assumed, not verified",
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
