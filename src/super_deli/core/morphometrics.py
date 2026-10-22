"""Fixed-landmark observations retaining raw geometry, identity and calibration metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from super_deli.core.data import Level

Coordinates = tuple[tuple[float | None, ...], ...]


@dataclass(frozen=True)
class LandmarkRecord:
    observation_id: str
    unit_id: str
    coordinates: Coordinates
    source_landmark_order: tuple[str, ...]
    scale_factor: float | None = None
    image: str | None = None
    tps_id: str | None = None
    comments: tuple[str, ...] = ()

    @property
    def missing_mask(self) -> tuple[tuple[bool, ...], ...]:
        return tuple(tuple(value is None for value in point) for point in self.coordinates)

    def to_dict(self) -> dict:
        return {**asdict(self), "missing_mask": self.missing_mask}


@dataclass(frozen=True)
class LandmarkCollection:
    modality_name: str
    level: Level
    dimensions: int
    landmark_ids: tuple[str, ...]
    coordinate_unit: str | None
    scale_unit: str | None
    records: tuple[LandmarkRecord, ...]

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            **asdict(self),
            "records": [record.to_dict() for record in self.records],
        }
