"""Fixed-landmark ingestion, GPA/tangent PCA and frozen-reference projection."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, replace
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

from super_deli.core.data import DatasetConfig
from super_deli.core.morphometrics import LandmarkCollection
from super_deli.core.provenance import configuration_hash
from super_deli.core.schema import config_from_dict
from super_deli.core.validation import validate_dataset
from super_deli.io.config import write_json
from super_deli.io.landmark_export import write_landmark_coordinates
from super_deli.io.morphometric import ingest_morphometric
from super_deli.models.shape_config import ShapeConfig

if TYPE_CHECKING:
    from super_deli.models.shape_pca import ShapePCAModel, ShapeProjection


@dataclass(frozen=True)
class ShapeAnalysisData:
    aligned: LandmarkCollection
    model: ShapePCAModel
    projection: ShapeProjection
    output_dir: Path
    summary: dict
    retained: LandmarkCollection


def run_shape_pca(
    dataset: DatasetConfig,
    *,
    model_config: ShapeConfig,
    base_dir: str | Path,
    output_dir: str | Path,
    modality_name: str | None = None,
    reference_dir: str | Path | None = None,
) -> ShapeAnalysisData:
    dataset = config_from_dict(dataset.to_dict())
    model_config = ShapeConfig.from_dict(model_config.to_dict())
    report = validate_dataset(dataset, base_dir=base_dir, check_files=False)
    if not report.valid:
        raise ValueError("; ".join(i.message for i in report.issues if i.severity == "error"))
    candidates = [
        m
        for m in dataset.modalities
        if m.kind == "morphometric" and m.format in {"tps", "landmarks_csv"}
    ]
    selected = [m for m in candidates if modality_name is None or m.name == modality_name]
    if len(selected) != 1:
        raise ValueError("Select exactly one TPS/landmark CSV modality with --modality")
    source = selected[0]
    try:
        np = import_module("numpy")
    except (ImportError, OSError) as exc:
        raise RuntimeError("Shape PCA requires: pip install 'super-deli[shape]'") from exc
    from super_deli.io.shape_model import read_shape_model, write_shape_model
    from super_deli.models.shape_pca import fit_shape_pca, project_shape_pca

    model, metadata, reference_inputs = None, None, []
    if reference_dir is not None:
        model, metadata, reference_inputs = read_shape_model(reference_dir)
        if model.config != model_config:
            raise ValueError("Projection configuration must match the saved training model")
        options = source.morphometric_options
        if (
            options is None
            or list(options.landmark_ids) != metadata["landmark_ids"]
            or options.dimensions != metadata["dimensions"]
        ):
            raise ValueError(
                "Projection landmark identities/order/dimensions differ from the model"
            )
    destination = Path(output_dir).absolute()
    destination.mkdir(exist_ok=False)
    try:
        loaded = ingest_morphometric(
            dataset,
            base_dir=base_dir,
            output_dir=destination / "ingestion",
            modality_names=[source.name],
        )
        collection = loaded.collections[0]
        records, exclusions = [], []
        for record in collection.records:
            if any(any(point) for point in record.missing_mask):
                if model_config.missing_policy == "error":
                    raise ValueError(
                        f"{record.observation_id}: incomplete landmarks; no imputation is performed"
                    )
                exclusions.append(
                    {
                        "observation_id": record.observation_id,
                        "unit_id": record.unit_id,
                        "reason": "incomplete_landmark_coordinates",
                    }
                )
            else:
                records.append(record)
        minimum = 2 if model is None else 1
        if len(records) < minimum:
            raise ValueError(f"At least {minimum} complete observations are required")
        repeats = {unit: n for unit, n in Counter(r.unit_id for r in records).items() if n > 1}
        if repeats and model_config.repeated_policy == "error":
            raise ValueError(
                "Repeated units require repeated_policy='allow'; observations are not aggregated"
            )
        coordinates = [r.coordinates for r in records]
        history = ()
        if model is None:
            fit = fit_shape_pca(coordinates, config=model_config)
            model, projection, history = fit.model, fit.projection, fit.history
            training_ids, training_units = (
                tuple(r.observation_id for r in records),
                tuple(r.unit_id for r in records),
            )
        else:
            projection = project_shape_pca(coordinates, model=model)
            training_ids, training_units = (
                tuple(metadata["training_observation_ids"]),
                tuple(metadata["training_unit_ids"]),
            )
        saved_model = write_shape_model(
            model,
            directory=destination,
            landmark_ids=collection.landmark_ids,
            observation_ids=training_ids,
            unit_ids=training_units,
        )
        aligned = replace(
            collection,
            coordinate_unit="unit_centroid_size",
            scale_unit=None,
            records=tuple(
                replace(
                    r,
                    coordinates=tuple(map(tuple, points.tolist())),
                    scale_factor=None,
                    comments=(*r.comments, "GPA coordinates; source scale not applied"),
                )
                for r, points in zip(records, projection.aligned, strict=True)
            ),
        )
        write_landmark_coordinates(
            aligned,
            dataset=dataset,
            source=source,
            directory=destination,
            stem="aligned_landmarks",
            config_name="aligned_dataset.json",
        )
        with (destination / "shape_arrays.npz").open("xb") as handle:
            np.savez(
                handle,
                aligned=projection.aligned,
                centroids=projection.centroids,
                centroid_sizes=projection.centroid_sizes,
                rotations=projection.rotations,
                tangent_coordinates=projection.tangent_coordinates,
                scores=projection.scores,
                angles_degrees=projection.angles_degrees,
                radial_shrinkage=projection.radial_shrinkage,
                reconstruction_error=projection.reconstruction_error,
            )
        with (destination / "scores.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                (
                    "observation_id",
                    "unit_id",
                    "centroid_size",
                    "angle_degrees",
                    "radial_shrinkage",
                    "pca_reconstruction_error",
                    *(f"PC{i + 1}" for i in range(projection.scores.shape[1])),
                )
            )
            for i, record in enumerate(records):
                writer.writerow(
                    (
                        record.observation_id,
                        record.unit_id,
                        projection.centroid_sizes[i],
                        projection.angles_degrees[i],
                        projection.radial_shrinkage[i],
                        projection.reconstruction_error[i],
                        *projection.scores[i],
                    )
                )
        write_json(history, destination / "gpa_history.json")
        configuration = {
            "dataset": dataset.to_dict(),
            "model": model_config.to_dict(),
            "modality": source.name,
            "reference_inputs": reference_inputs,
        }
        summary = {
            "schema_version": 1,
            "status": "complete",
            "mode": "fit" if reference_dir is None else "project",
            "dataset_id": dataset.dataset_id,
            "modality": source.name,
            "level": collection.level,
            "observation_ids": [r.observation_id for r in records],
            "unit_ids": [r.unit_id for r in records],
            "excluded_observations": exclusions,
            "repeated_units": repeats,
            "landmark_ids": collection.landmark_ids,
            "dimensions": collection.dimensions,
            "gpa": {
                "reference": "normalized consensus",
                "reference_orientation": "first retained training observation",
                "reflection": model_config.reflection,
                "iterations": len(history),
                "convergence": "converged" if reference_dir is None else "frozen_reference",
                "size": "unit centroid size; raw source centroid sizes retained",
                "source_coordinate_unit": collection.coordinate_unit,
                "source_scale_application": "none",
            },
            "pca": {
                "status": "available" if model.numerical_rank else "no_resolved_shape_variation",
                "projection": "orthogonal tangent residuals at the training consensus",
                "scaling": "centered covariance PCA; no coordinate standardization or whitening",
                "component_order": "landmark major, x/y/z",
                "numerical_rank": model.numerical_rank,
                "shape_dimension": model.shape_dimension,
                "retained_components": model.components.shape[0],
                "requested_components": model_config.n_components,
                "total_variance": model.total_variance,
                "explained_variance": model.explained_variance.tolist(),
                "explained_variance_ratio": model.explained_variance_ratio.tolist(),
            },
            "diagnostics": {
                "max_consensus_angle_degrees": float(projection.angles_degrees.max()),
                "max_orthographic_radial_shrinkage": float(projection.radial_shrinkage.max()),
                "reflected_observation_ids": [
                    r.observation_id
                    for r, determinant in zip(
                        records, np.linalg.det(projection.rotations), strict=True
                    )
                    if determinant < 0
                ],
                "training_unit_overlap": sorted({r.unit_id for r in records} & set(training_units))
                if reference_dir is not None
                else [],
            },
            "uncertainty": {
                "status": "not_computed",
                "reason": "Descriptive ordination without resampling, null model or species test",
            },
            "interpretation": [
                "Fixed landmark correspondence and consistent axes are input assumptions",
                "Every observation has equal weight; repeated units are not independent replicates",
                "Tangent PCA is local; inspect angular/radial distortion diagnostics",
                "No species inference, allometry correction or semilandmark sliding",
            ],
            "model_array_sha256": saved_model["array_sha256"],
            "provenance": {
                "configuration": configuration,
                "configuration_sha256": configuration_hash(configuration),
                "ingestion": loaded.summary["provenance"],
                "software": {**loaded.summary["provenance"]["software"], "numpy": version("numpy")},
            },
        }
        write_json(summary, destination / "summary.json")
        return ShapeAnalysisData(
            aligned,
            model,
            projection,
            destination,
            summary,
            replace(collection, records=tuple(records)),
        )
    except Exception as exc:
        if hasattr(exc, "history"):
            write_json(exc.history, destination / "gpa_history.json")
        write_json({"status": "failed", "error": str(exc)}, destination / "failure.json")
        raise
