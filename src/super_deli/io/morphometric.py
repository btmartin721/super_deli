"""Orchestrate TPS/coordinate CSV ingestion with source preservation and provenance."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from super_deli.core.data import DatasetConfig
from super_deli.core.morphometrics import LandmarkCollection
from super_deli.core.provenance import collect_provenance
from super_deli.core.schema import config_from_dict
from super_deli.core.validation import resolve_input, validate_dataset
from super_deli.io.config import write_json
from super_deli.io.landmark_csv import read_landmark_csv
from super_deli.io.morphometric_base import MorphometricIngestionError, check_coverage, options_for
from super_deli.io.staging import stage_input
from super_deli.io.tps import read_tps


@dataclass(frozen=True)
class LoadedMorphometricData:
    collections: tuple[LandmarkCollection, ...]
    output_dir: Path
    summary: dict


def _summary(collection: LandmarkCollection) -> dict:
    missing, partial, absent_scales, all_missing = {}, {}, [], []
    for record in collection.records:
        mask = record.missing_mask
        missing[record.observation_id] = sum(all(point) for point in mask)
        partial[record.observation_id] = sum(any(point) and not all(point) for point in mask)
        if missing[record.observation_id] == len(collection.landmark_ids):
            all_missing.append(record.observation_id)
        if record.scale_factor is None:
            absent_scales.append(record.observation_id)
    return {
        "missing_landmarks_by_observation": missing,
        "partial_landmarks_by_observation": partial,
        "all_missing_observation_ids": all_missing,
        "missing_scale_observation_ids": absent_scales,
    }


def ingest_morphometric(
    config: DatasetConfig,
    *,
    base_dir: str | Path,
    output_dir: str | Path,
    modality_names: Sequence[str] | None = None,
) -> LoadedMorphometricData:
    """Load fixed landmarks without image decoding, rescaling, alignment or inference.

    Default to all TPS/landmark CSV modalities. Image manifests remain declarations.
    A new directory contains staged sources, geometry JSON, and a completion summary;
    failures after creation retain diagnostics without a success summary.
    """
    config = config_from_dict(config.to_dict())
    report = validate_dataset(config, base_dir=base_dir, check_files=False)
    if not report.valid:
        raise MorphometricIngestionError(
            "; ".join(f"{i.location}: {i.message}" for i in report.issues if i.severity == "error")
        )
    supported = {
        m.name: m
        for m in config.modalities
        if m.kind == "morphometric" and m.format in {"tps", "landmarks_csv"}
    }
    if modality_names is not None:
        if (
            isinstance(modality_names, str)
            or not isinstance(modality_names, Sequence)
            or not modality_names
            or any(not isinstance(n, str) for n in modality_names)
        ):
            raise MorphometricIngestionError("modality_names must be a nonempty sequence of names")
        if len(set(modality_names)) != len(modality_names):
            raise MorphometricIngestionError("Duplicate requested modality names")
        if unknown := set(modality_names) - supported.keys():
            raise MorphometricIngestionError(
                f"Unknown or unsupported landmark modalities: {sorted(unknown)}; "
                "select TPS or landmark CSV"
            )
    selected = tuple(
        m for name, m in supported.items() if modality_names is None or name in modality_names
    )
    if not selected:
        raise MorphometricIngestionError(
            "Declare at least one TPS or landmark CSV source; image manifests are not decoded"
        )
    sources = []
    for modality in selected:
        options_for(modality)
        source = resolve_input(modality.path, base_dir)
        if not source.is_file():
            raise MorphometricIngestionError(f"Not a regular input file: {source}")
        sources.append(source)
    destination = Path(output_dir).absolute()
    destination.mkdir(exist_ok=False)
    try:
        collections, summaries, fingerprints = [], [], []
        coverage = {c.modality: asdict(c) for c in report.coverage}
        for i, (modality, source) in enumerate(zip(selected, sources, strict=True)):
            directory = destination / f"modality_{i:03d}"
            directory.mkdir()
            staged, fingerprint = stage_input(source, directory / "input")
            fingerprints.append(fingerprint)
            reader = read_tps if modality.format == "tps" else read_landmark_csv
            records = reader(staged, modality)
            check_coverage(records, modality)
            options = options_for(modality)
            collection = LandmarkCollection(
                modality.name,
                modality.level,
                options.dimensions,
                options.landmark_ids,
                options.coordinate_unit,
                options.scale_unit,
                records,
            )
            artifact = f"modality_{i:03d}/landmarks.json"
            write_json(collection.to_dict(), destination / artifact)
            collections.append(collection)
            summaries.append(
                {
                    "name": modality.name,
                    "level": modality.level,
                    "source_format": modality.format,
                    "artifact": artifact,
                    "dimensions": options.dimensions,
                    "landmark_ids": options.landmark_ids,
                    "reader_options": asdict(options),
                    "coverage": coverage[modality.name],
                    **_summary(collection),
                }
            )
        provenance = collect_provenance(config, base_dir=base_dir)
        provenance.update(input_hash_status="computed_selected_modalities", inputs=fingerprints)
        summary = {
            "schema_version": 1,
            "status": "complete",
            "dataset_id": config.dataset_id,
            "modalities": summaries,
            "provenance": provenance,
            "processing": {
                "coordinate_order": "declared_landmark_ids",
                "scale_application": "none",
                "image_decoding": "none",
                "imputation": "none",
                "procrustes_alignment": "none",
            },
        }
        write_json(summary, destination / "summary.json")
        return LoadedMorphometricData(tuple(collections), destination, summary)
    except Exception as exc:
        write_json({"status": "failed", "error": str(exc)}, destination / "failure.json")
        raise
