"""Strict, dependency-free rules for locating named visible image markers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields


def _integer(value: object, name: str, low: int, high: int | None = None) -> None:
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError(f"{name} must be an integer in [{low}, {high or 'unbounded'}]")


def _object(value: object, cls: type, required: set[str]) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{cls.__name__} must be an object")
    unknown = set(value) - {f.name for f in fields(cls)}
    missing = required - value.keys()
    if unknown or missing:
        raise ValueError(f"{cls.__name__}: unknown={sorted(unknown)}; missing={sorted(missing)}")
    return dict(value)


@dataclass(frozen=True)
class MarkerRule:
    landmark_id: str
    rgb_lower: tuple[int, int, int]
    rgb_upper: tuple[int, int, int]
    min_area: int
    max_area: int
    search_box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

    def __post_init__(self):
        name = self.landmark_id
        if not isinstance(name, str) or not name or name != name.strip() or "\x00" in name:
            raise ValueError("landmark_id must be a nonempty, unpadded identifier without NUL")
        for name in ("rgb_lower", "rgb_upper"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(values) != 3:
                raise ValueError(f"{name} must contain three RGB integers")
            for value in values:
                _integer(value, name, 0, 255)
        if any(a > b for a, b in zip(self.rgb_lower, self.rgb_upper, strict=True)):
            raise ValueError("rgb_lower must not exceed rgb_upper")
        _integer(self.min_area, "min_area", 1)
        _integer(self.max_area, "max_area", self.min_area)
        box = self.search_box
        if not isinstance(box, tuple) or len(box) != 4:
            raise ValueError("search_box requires [left, top, right, bottom]")
        if any(type(v) not in (float, int) or not 0 <= v <= 1 or not math.isfinite(v) for v in box):
            raise ValueError("search_box values must be finite normalized numbers in [0, 1]")
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("search_box must have positive width and height")

    @classmethod
    def from_dict(cls, value: object) -> MarkerRule:
        data = _object(
            value, cls, {"landmark_id", "rgb_lower", "rgb_upper", "min_area", "max_area"}
        )
        for name in ("rgb_lower", "rgb_upper", "search_box"):
            if name in data:
                if not isinstance(data[name], list):
                    raise ValueError(f"{name} must be an array")
                data[name] = tuple(data[name])
        return cls(**data)


@dataclass(frozen=True)
class MarkerDetectorConfig:
    landmarks: tuple[MarkerRule, ...]
    connectivity: int = 8
    min_alpha: int = 255
    max_pixels: int = 4000000
    max_components: int = 10000
    schema_version: int = 1
    method: str = "rgb_components"

    def __post_init__(self):
        _integer(self.schema_version, "schema_version", 1, 1)
        if self.method != "rgb_components":
            raise ValueError("Only method='rgb_components' is supported")
        _integer(self.connectivity, "connectivity", 4, 8)
        if self.connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        _integer(self.min_alpha, "min_alpha", 1, 255)
        _integer(self.max_pixels, "max_pixels", 1)
        _integer(self.max_components, "max_components", 1)
        if (
            not isinstance(self.landmarks, tuple)
            or not 1 <= len(self.landmarks) <= 128
            or any(not isinstance(r, MarkerRule) for r in self.landmarks)
        ):
            raise ValueError("landmarks must contain 1 through 128 MarkerRule objects")
        ids = [r.landmark_id for r in self.landmarks]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate landmark IDs")
        # A pixel must never be eligible for two named landmarks. Identity comes
        # from declared color/space rules, not component order or relative size.
        for i, left in enumerate(self.landmarks):
            for right in self.landmarks[i + 1 :]:
                colors_overlap = all(
                    max(a, c) <= min(b, d)
                    for a, b, c, d in zip(
                        left.rgb_lower,
                        left.rgb_upper,
                        right.rgb_lower,
                        right.rgb_upper,
                        strict=True,
                    )
                )
                a, b = left.search_box, right.search_box
                space_overlaps = max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(
                    a[3], b[3]
                )
                if colors_overlap and space_overlaps:
                    raise ValueError(
                        f"Overlapping color/search rules for {left.landmark_id!r} and "
                        f"{right.landmark_id!r}; make landmark identity disjoint"
                    )

    def to_dict(self) -> dict:
        import json

        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, value: object) -> MarkerDetectorConfig:
        data = _object(value, cls, {"schema_version", "method", "landmarks"})
        if not isinstance(data["landmarks"], list):
            raise ValueError("landmarks must be an array")
        data["landmarks"] = tuple(MarkerRule.from_dict(r) for r in data["landmarks"])
        return cls(**data)
