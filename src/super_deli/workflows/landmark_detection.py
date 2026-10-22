"""Connect image ingestion to explicit visible-marker detection and auditable exports."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from super_deli.core.data import DatasetConfig
from super_deli.core.morphometrics import LandmarkCollection, LandmarkRecord
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.core.validation import validate_dataset
from super_deli.io.config import write_json
from super_deli.io.detection_artifacts import write_detected_coordinates, write_detection_overlay
from super_deli.io.image_decode import image_dependencies
from super_deli.io.images import ingest_images
from super_deli.models.marker_config import MarkerDetectorConfig
from super_deli.models.marker_detection import detect_markers, detector_dependencies


@dataclass(frozen=True)
class DetectedLandmarkData:
    collection: LandmarkCollection
    output_dir: Path
    summary: dict


def run_landmark_detection(
    dataset: DatasetConfig,
    *,
    detector_config: MarkerDetectorConfig,
    base_dir: str | Path,
    output_dir: str | Path,
    modality_name: str | None = None,
) -> DetectedLandmarkData:
    """Detect named visible markers in one image modality, with unresolved nulls.

    The existing image reader owns orientation/page/frame handling. Detection
    works only in that derivative's raster frame, and every centroid is unreviewed.
    A completed run may contain unresolved markers; counts remain explicit.
    """
    dataset = config_from_dict(dataset.to_dict())
    detector_config = MarkerDetectorConfig.from_dict(detector_config.to_dict())
    report = validate_dataset(dataset, base_dir=base_dir, check_files=False)
    if not report.valid:
        raise ValueError("; ".join(i.message for i in report.issues if i.severity == "error"))
    available = [
        m for m in dataset.modalities if m.kind == "morphometric" and m.format == "image_manifest"
    ]
    selected = [m for m in available if modality_name is None or m.name == modality_name]
    if len(selected) != 1:
        raise ValueError("Select exactly one declared image_manifest modality with --modality")
    modality = selected[0]
    software = {**detector_dependencies(), **image_dependencies()}
    from PIL import Image

    destination = Path(output_dir).absolute()
    destination.mkdir(exist_ok=False)
    try:
        loaded = ingest_images(
            dataset,
            base_dir=base_dir,
            output_dir=destination / "ingestion",
            modality_names=[modality.name],
        )
        software = {**loaded.summary["provenance"]["software"], **software}
        (destination / "overlays").mkdir()
        records, diagnostics, counts = [], [], Counter()
        for index, record in enumerate(loaded.collections[0].records):
            image_ref = "ingestion/" + record.image_ref
            image_path = destination / image_ref
            if fingerprint_file(image_path).sha256 != record.image_sha256:
                raise OSError("Display derivative changed before detection")
            overlay_ref = f"overlays/image_{index:03d}.png"
            with Image.open(image_path, formats=("PNG",)) as image:
                if image.size != (record.width, record.height) or image.mode != record.mode:
                    raise ValueError("Display derivative dimensions/mode disagree with ingestion")
                detections = detect_markers(image, config=detector_config)
                write_detection_overlay(image, detections, destination / overlay_ref)
            if fingerprint_file(image_path).sha256 != record.image_sha256:
                raise OSError("Display derivative changed during detection")
            counts.update(d.status for d in detections)
            records.append(
                LandmarkRecord(
                    observation_id=record.observation_id,
                    unit_id=record.unit_id,
                    coordinates=tuple(d.coordinate for d in detections),
                    source_landmark_order=tuple(d.landmark_id for d in detections),
                    image=image_ref,
                    comments=(
                        "Unreviewed visible-marker centroids in the display derivative frame",
                    ),
                )
            )
            diagnostics.append(
                {
                    "observation_id": record.observation_id,
                    "unit_id": record.unit_id,
                    "image_ref": image_ref,
                    "image_sha256": record.image_sha256,
                    "width": record.width,
                    "height": record.height,
                    "overlay_ref": overlay_ref,
                    "overlay_sha256": fingerprint_file(destination / overlay_ref).sha256,
                    "landmarks": [d.to_dict() for d in detections],
                }
            )
        names = tuple(r.landmark_id for r in detector_config.landmarks)
        collection = LandmarkCollection(
            modality.name, modality.level, 2, names, "pixel", None, tuple(records)
        )
        write_detected_coordinates(
            collection, dataset=dataset, source=modality, directory=destination
        )
        write_json(
            {"schema_version": 1, "landmark_ids": names, "records": diagnostics},
            destination / "detections.json",
        )
        configuration = {
            "dataset": dataset.to_dict(),
            "detector": detector_config.to_dict(),
            "modality": modality.name,
        }
        summary = {
            "schema_version": 1,
            "status": "complete",
            "dataset_id": dataset.dataset_id,
            "modality": modality.name,
            "method": detector_config.method,
            "observation_count": len(records),
            "landmark_ids": names,
            "detection_counts": dict(counts),
            "unresolved_count": sum(v for k, v in counts.items() if k != "detected"),
            "review_status": "unreviewed",
            "coordinate_frame": {
                "origin": "top_left",
                "x_direction": "right",
                "y_direction": "down",
                "unit": "pixel",
                "pixel_centers": "integer indices",
                "alignment": "none",
            },
            "localization_uncertainty": {
                "status": "not_computed",
                "reason": "No annotated-image calibration; component geometry is not confidence",
            },
            "artifacts": {
                "coordinates": "landmarks.json",
                "csv": "landmarks.csv",
                "dataset": "landmark_dataset.json",
                "diagnostics": "detections.json",
            },
            "provenance": {
                "configuration": configuration,
                "configuration_sha256": configuration_hash(configuration),
                "ingestion": loaded.summary["provenance"],
                "software": software,
            },
            "interpretation": [
                "Visible-marker ink centroids are provisional and require review",
                "Rules define identity; no anatomical model, OCR or TPS extraction is performed",
                "No scaling, shape alignment, missing-point imputation or species inference",
            ],
        }
        write_json(summary, destination / "summary.json")
        return DetectedLandmarkData(collection, destination, summary)
    except Exception as exc:
        write_json({"status": "failed", "error": str(exc)}, destination / "failure.json")
        raise
