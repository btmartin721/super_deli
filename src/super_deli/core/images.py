"""Image observations describe derivatives; pixels stay in separate PNG artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from super_deli.core.data import Level


@dataclass(frozen=True)
class ImageRecord:
    observation_id: str
    unit_id: str
    source_ref: str
    source_sha256: str
    image_ref: str
    image_sha256: str
    source_format: str
    page: int | None
    frame: int | None
    width: int
    height: int
    mode: str
    decoding: dict
    manifest_metadata: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ImageCollection:
    modality_name: str
    level: Level
    records: tuple[ImageRecord, ...]

    def to_dict(self) -> dict:
        return {"schema_version": 1, **asdict(self)}
