"""Explicit assumptions for fitted Gaussian nulls in continuous raw views."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

from super_deli.models.integrative_config import IntegrativeConfig, configuration_dict

CASES = ("spherical_gaussian", "correlated_gaussian", "complementary", "heavy_tailed")


def _options(cls, value, required):
    if (
        not isinstance(value, dict)
        or value.keys() - {f.name for f in fields(cls)}
        or required - value.keys()
    ):
        raise ValueError(f"Invalid {cls.__name__}; required fields: {sorted(required)}")
    return dict(value)


@dataclass(frozen=True)
class EmpiricalCalibrationConfig:
    assume_iid_gaussian_raw_views: bool
    model: IntegrativeConfig
    n_simulations: int = 99
    alpha: float = 0.05
    random_seed: int = 20261023
    rank_tol: float = 1e-12
    max_archive_bytes: int = 2147483648
    schema_version: int = 1

    def __post_init__(self):
        if self.assume_iid_gaussian_raw_views is not True or not isinstance(
            self.model, IntegrativeConfig
        ):
            raise ValueError("Explicitly assume iid joint Gaussian raw continuous views")
        if self.model.covariance_type != "full":
            raise ValueError("Shared-covariance significance calibration is not yet supported")
        for name in ("n_simulations", "max_archive_bytes", "schema_version"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.schema_version != 1
            or type(self.random_seed) is not int
            or not 0 <= self.random_seed < 2**64
        ):
            raise ValueError("Use schema version 1 and a uint64 random seed")
        for name in ("alpha", "rank_tol"):
            value = getattr(self, name)
            try:
                valid = type(value) in (int, float) and math.isfinite(value) and 0 < value < 1
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(f"{name} must be finite and strictly between zero and one")
        m = self.model
        if (
            len(m.views) != 2
            or {v.kind for v in m.views} != {"morphological", "ecological"}
            or m.target_level != "specimen"
            or m.missing_views != "error"
            or m.n_bootstrap != 0
            or m.max_k < 2
            or any(
                v.aggregation != "none"
                or v.missing_policy != "error"
                or v.count_transform != "identity"
                for v in m.views
            )
        ):
            raise ValueError(
                "Use matched morphology/ecology; no aggregation, missingness or bootstrap; K>=2"
            )

    @classmethod
    def from_dict(cls, value):
        options = _options(cls, value, {"schema_version", "model", "assume_iid_gaussian_raw_views"})
        options["model"] = IntegrativeConfig.from_dict(options["model"])
        return cls(**options)

    def to_dict(self):
        return configuration_dict(self)


@dataclass(frozen=True)
class EmpiricalStudyConfig:
    calibration: EmpiricalCalibrationConfig
    n_datasets: int = 25
    n_units: int = 64
    random_seed: int = 20261023
    schema_version: int = 1

    def __post_init__(self):
        if not isinstance(self.calibration, EmpiricalCalibrationConfig):
            raise ValueError("An empirical calibration policy is required")
        validate_study_dimensions(self)

    @classmethod
    def from_dict(cls, value):
        options = _options(cls, value, {"schema_version", "calibration"})
        options["calibration"] = EmpiricalCalibrationConfig.from_dict(options["calibration"])
        return cls(**options)

    def to_dict(self):
        return configuration_dict(self)


def validate_study_dimensions(study):
    """Shared independent-dataset, seed and workspace contract."""
    for name in ("n_datasets", "n_units", "schema_version"):
        if type(getattr(study, name)) is not int or getattr(study, name) < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (
        study.schema_version != 1
        or study.n_units < 20
        or study.n_units % 4
        or type(study.random_seed) is not int
        or not 0 <= study.random_seed < 2**64
    ):
        raise ValueError("Use schema 1, >=20 units divisible by four and a uint64 seed")
    if 128 * study.n_units * 64 > study.calibration.model.max_working_bytes:
        raise ValueError("Study dimensions exceed workspace budget")
