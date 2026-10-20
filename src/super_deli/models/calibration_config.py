"""Explicit assumptions for a conditional, approximate Gaussian score test."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class GaussianCalibrationConfig:
    assume_iid_gaussian_scores: bool
    n_simulations: int = 199
    alpha: float = 0.05
    random_seed: int = 0
    rank_tol: float = 1e-12
    max_working_bytes: int = 268435456
    schema_version: int = 1

    def __post_init__(self):
        if self.assume_iid_gaussian_scores is not True:
            raise ValueError("Explicitly assume iid Gaussian target scores under the null")
        for name in ("n_simulations", "max_working_bytes", "schema_version"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError("random_seed must be a nonnegative integer")
        if self.schema_version != 1:
            raise ValueError("Unsupported Gaussian calibration configuration version")
        for name in ("alpha", "rank_tol"):
            value = getattr(self, name)
            try:
                valid = type(value) in (int, float) and math.isfinite(value) and 0 < value < 1
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(f"{name} must be finite and strictly between zero and one")

    @classmethod
    def from_dict(cls, value):
        if (
            not isinstance(value, dict)
            or value.keys() - {f.name for f in fields(cls)}
            or not {"schema_version", "assume_iid_gaussian_scores"} <= value.keys()
        ):
            raise ValueError("Calibration requires version, iid assumption and supported fields")
        return cls(**value)

    def to_dict(self):
        return asdict(self)
