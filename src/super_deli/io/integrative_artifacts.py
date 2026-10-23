"""Numerical integration parameters and explicit source-to-target membership exports."""

from __future__ import annotations

from pathlib import Path

from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import write_json


def write_pca_parameters(pca, path: Path, **extra):
    import numpy as np

    with path.open("xb") as handle:
        np.savez(
            handle,
            mean=pca.mean_,
            components=pca.components_,
            explained_variance=pca.explained_variance_,
            explained_variance_ratio=pca.explained_variance_ratio_,
            singular_values=pca.singular_values_,
            **extra,
        )


def write_integration_parameters(fit, destination):
    import numpy as np

    from super_deli.io.shape_model import write_shape_model

    summaries = []
    for i, (data, fitted) in enumerate(zip(fit.inputs, fit.pipeline.views, strict=True)):
        directory = destination / f"view_{i:03d}" / "parameters"
        directory.mkdir()
        with (directory / "profiles.npz").open("xb") as handle:
            np.savez(
                handle,
                profiles=fitted.profiles,
                mean=fitted.mean,
                scale=fitted.scale,
                normalized_weight=fitted.normalized_weight,
            )
        members = [
            {
                "target_id": target,
                "observation_ids": [
                    o
                    for o, t in zip(data.observation_ids, data.target_ids, strict=True)
                    if t == target
                ],
                "source_unit_ids": [
                    u for u, t in zip(data.unit_ids, data.target_ids, strict=True) if t == target
                ],
            }
            for target in fit.unit_ids
        ]
        if data.kind == "morphometric":
            write_shape_model(
                fitted.model,
                directory=directory,
                landmark_ids=data.landmark_ids,
                observation_ids=tuple(data.observation_ids[j] for j in fitted.training_indices),
                unit_ids=tuple(data.unit_ids[j] for j in fitted.training_indices),
            )
        else:
            if data.kind == "genomic":
                preprocessing, pca = fitted.model
            else:
                preprocessing, pca = fitted.model, fitted.model.pca
            write_pca_parameters(
                pca,
                directory / "pca.npz",
                kept_indices=preprocessing.kept_indices,
                feature_means=preprocessing.means,
                feature_scales=preprocessing.scales,
            )
        summary = {
            "schema_version": 1,
            "modality": data.name,
            "kind": data.kind,
            "source_level": data.level,
            "aggregation": fitted.config.aggregation,
            "profile_definition": "Arithmetic mean of source PC scores within each target",
            "feature_names": data.feature_names,
            "feature_types": data.feature_types,
            "landmark_ids": data.landmark_ids,
            "normalized_weight": fitted.normalized_weight,
            "score_scale": fitted.scale,
            "n_pcs": fitted.config.n_pcs,
            "source_fit_weighting": (
                "Equal target means per draw; repeated draws retain their multiplicity"
                if fitted.config.pca_fit == "target_means"
                else "Equal source observations; target draws retain all members"
            ),
            "retained_source_indices": list(data.original_indices),
            "training_source_indices": [data.original_indices[j] for j in fitted.training_indices],
            "bootstrap_source_index_basis": "Retained rows; map through retained_source_indices",
            "members": members,
            "diagnostics": fitted.diagnostics,
            "arrays": {p.name: fingerprint_file(p).sha256 for p in sorted(directory.glob("*.npz"))},
        }
        write_json(summary, directory / "summary.json")
        summaries.append(summary)
    write_pca_parameters(fit.pipeline.pca, destination / "fusion.npz", fused=fit.pipeline.fused)
    return summaries
