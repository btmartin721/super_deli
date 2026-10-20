"""Visible-marker diagnostics retain unresolved identity and provisional coordinates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class MarkerCandidate:
    pixel_count: int
    centroid: tuple[float, float]
    bounding_box: tuple[int, int, int, int]
    touches_search_boundary: bool


@dataclass(frozen=True)
class LandmarkDetection:
    landmark_id: str
    status: Literal["detected", "missing", "ambiguous", "rejected", "boundary"]
    coordinate: tuple[float | None, float | None]
    reason: str
    search_box_pixels: tuple[int, int, int, int]
    component_count: int
    area_rejected_count: int
    candidates: tuple[MarkerCandidate, ...]

    def to_dict(self) -> dict:
        return asdict(self)
