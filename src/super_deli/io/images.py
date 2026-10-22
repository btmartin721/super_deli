"""Image-manifest ingestion with stable input copies and auditable PNG derivatives."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from super_deli.core.data import DatasetConfig, ImageReaderOptions
from super_deli.core.images import ImageCollection, ImageRecord
from super_deli.core.provenance import collect_provenance, fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.core.validation import resolve_input, validate_dataset
from super_deli.io.config import write_json
from super_deli.io.image_decode import decode_image, image_dependencies
from super_deli.io.image_manifest import ImageIngestionError, read_image_manifest
from super_deli.io.staging import stage_input


@dataclass(frozen=True)
class LoadedImageData:
    collections: tuple[ImageCollection, ...]
    output_dir: Path
    summary: dict


def ingest_images(
    config: DatasetConfig,
    *,
    base_dir: str | Path,
    output_dir: str | Path,
    modality_names: Sequence[str] | None = None,
) -> LoadedImageData:
    """Decode declared image observations; keep biological links and raw inputs intact.

    Image paths resolve against the original manifest's parent directory. Source
    assets shared by several page/frame observations are staged once per modality.
    Landmark/TPS references and group/split columns remain unprocessed metadata.
    """
    config = config_from_dict(config.to_dict())
    report = validate_dataset(config, base_dir=base_dir, check_files=False)
    if not report.valid:
        raise ImageIngestionError(
            "; ".join(f"{i.location}: {i.message}" for i in report.issues if i.severity == "error")
        )
    supported = {
        m.name: m
        for m in config.modalities
        if m.kind == "morphometric" and m.format == "image_manifest"
    }
    if modality_names is not None:
        if (
            isinstance(modality_names, str)
            or not isinstance(modality_names, Sequence)
            or not modality_names
            or any(not isinstance(n, str) for n in modality_names)
        ):
            raise ImageIngestionError("modality_names must be a nonempty sequence of names")
        if len(set(modality_names)) != len(modality_names):
            raise ImageIngestionError("Duplicate requested modality names")
        if unknown := set(modality_names) - supported.keys():
            raise ImageIngestionError(f"Unknown image_manifest modalities: {sorted(unknown)}")
    selected = tuple(
        m for name, m in supported.items() if modality_names is None or name in modality_names
    )
    if not selected:
        raise ImageIngestionError("Declare at least one image_manifest modality")
    sources = []
    for modality in selected:
        options = modality.image_options or ImageReaderOptions()
        source = resolve_input(modality.path, base_dir)
        if not source.is_file() or source.stat().st_size > options.max_source_bytes:
            raise ImageIngestionError(f"Missing or oversized image manifest: {source}")
        sources.append(source)
    software = image_dependencies()
    destination = Path(output_dir).absolute()
    destination.mkdir(exist_ok=False)
    try:
        collections, summaries, fingerprints = [], [], []
        coverage = {c.modality: asdict(c) for c in report.coverage}
        for i, (modality, source) in enumerate(zip(selected, sources, strict=True)):
            options = modality.image_options or ImageReaderOptions()
            directory = destination / f"modality_{i:03d}"
            directory.mkdir()
            manifest, manifest_fingerprint = stage_input(source, directory / "manifest")
            if manifest_fingerprint["size_bytes"] > options.max_source_bytes:
                raise ImageIngestionError("Image manifest exceeds max_source_bytes after staging")
            fingerprints.append(manifest_fingerprint)
            references = read_image_manifest(manifest, modality, source_dir=source.parent)
            (directory / "sources").mkdir()
            (directory / "images").mkdir()
            staged_sources, records = {}, []
            for j, reference in enumerate(references):
                original = reference.path
                if original not in staged_sources:
                    if not original.is_file() or original.stat().st_size > options.max_source_bytes:
                        raise ImageIngestionError(f"Missing or oversized image source: {original}")
                    staged, fingerprint = stage_input(
                        original, directory / "sources" / f"source_{len(staged_sources):03d}"
                    )
                    if fingerprint["size_bytes"] > options.max_source_bytes:
                        raise ImageIngestionError(
                            "Image source exceeds max_source_bytes after staging"
                        )
                    fingerprints.append(fingerprint)
                    staged_sources[original] = staged, fingerprint
                staged, fingerprint = staged_sources[original]
                raster, format, page, frame, decoding = decode_image(
                    staged, page=reference.page, frame=reference.frame, options=options
                )
                artifact = directory / "images" / f"image_{j:03d}.png"
                with raster:
                    width, height = raster.size
                    mode = raster.mode
                    with artifact.open("xb") as handle:
                        raster.save(handle, format="PNG")
                image_fingerprint = fingerprint_file(artifact)
                records.append(
                    ImageRecord(
                        reference.observation_id,
                        reference.unit_id,
                        str(staged.relative_to(destination)),
                        fingerprint["sha256"],
                        str(artifact.relative_to(destination)),
                        image_fingerprint.sha256,
                        format,
                        page,
                        frame,
                        width,
                        height,
                        mode,
                        decoding,
                        reference.metadata,
                    )
                )
            collection = ImageCollection(modality.name, modality.level, tuple(records))
            collection_ref = f"modality_{i:03d}/images.json"
            write_json(collection.to_dict(), destination / collection_ref)
            collections.append(collection)
            summaries.append(
                {
                    "name": modality.name,
                    "level": modality.level,
                    "artifact": collection_ref,
                    "coverage": coverage[modality.name],
                    "source_count": len(staged_sources),
                    "image_count": len(records),
                    "reader_options": asdict(options),
                }
            )
        provenance = collect_provenance(config, base_dir=base_dir)
        provenance.update(
            input_hash_status="computed_selected_manifests_and_images", inputs=fingerprints
        )
        provenance["software"].update(software)
        summary = {
            "schema_version": 1,
            "status": "complete",
            "dataset_id": config.dataset_id,
            "modalities": summaries,
            "provenance": provenance,
            "processing": {
                "derivative": "8-bit RGB/RGBA PNG",
                "landmark_detection": "none",
                "coordinate_alignment": "none",
                "scale_application": "none",
                "group_split_use": "metadata_only",
            },
        }
        write_json(summary, destination / "summary.json")
        return LoadedImageData(tuple(collections), destination, summary)
    except Exception as exc:
        write_json({"status": "failed", "error": str(exc)}, destination / "failure.json")
        raise
