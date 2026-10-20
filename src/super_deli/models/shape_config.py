"""Dependency-free policies for fixed-landmark GPA and tangent-space PCA."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class ShapeConfig:
    reflection: str = "forbid"
    missing_policy: str = "error"
    repeated_policy: str = "error"
    n_components: int | None = None
    max_iter: int = 500
    tol: float = 1e-10
    rank_tol: float = 1e-12
    max_working_bytes: int = 268435456
    landmark_type: str = "fixed"
    schema_version: int = 1

    def __post_init__(self):
        for name, choices in (
            ("reflection", {"forbid", "allow"}),
            ("missing_policy", {"error", "drop_observations"}),
            ("repeated_policy", {"error", "allow"}),
            ("landmark_type", {"fixed"}),
        ):
            if not isinstance(getattr(self, name), str) or getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {sorted(choices)}")
        for name in ("max_iter", "max_working_bytes", "schema_version"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.schema_version != 1:
            raise ValueError("Unsupported shape configuration version")
        if self.n_components is not None and (
            type(self.n_components) is not int or self.n_components < 1
        ):
            raise ValueError("n_components must be a positive integer or null")
        for name in ("tol", "rank_tol"):
            v = getattr(self, name)
            if type(v) not in (int, float) or not 0 < v < 1 or not math.isfinite(v):
                raise ValueError(f"{name} must be finite and strictly between 0 and 1")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> ShapeConfig:
        if not isinstance(value, dict):
            raise ValueError("Shape configuration must be an object")
        unknown = value.keys() - {f.name for f in fields(cls)}
        if unknown or "schema_version" not in value:
            raise ValueError(
                f"Shape configuration requires schema_version; unknown={sorted(unknown)}"
            )
        return cls(**value)
