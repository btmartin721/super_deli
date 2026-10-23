"""Explicit fixed-design assumptions for a hierarchical continuous-data null."""

from __future__ import annotations

import math
from dataclasses import dataclass

from super_deli.models.empirical_config import _options
from super_deli.models.integrative_config import (
    IntegrativeConfig,
    configuration_dict,
    require_calibrated_pca,
)

CASES = ("independent_gaussian", "correlated_gaussian", "complementary", "heavy_tailed")
BOUNDARY_CASES = ("zero_conditional_gaussian", "rank_two_gaussian", "weak_conditional_gaussian")


@dataclass(frozen=True)
class BoundarySolverConfig:
    estimator: str = "conditional_psd_v1"
    max_polish_iterations: int = 2000
    max_line_search: int = 40

    def __post_init__(self):
        if self.estimator != "conditional_psd_v1":
            raise ValueError("Unsupported grouped covariance estimator")
        for name in ("max_polish_iterations", "max_line_search"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_dict(cls, value):
        return cls(**_options(cls, value, {"estimator"}))


def study_cases(study):
    return CASES + BOUNDARY_CASES if study.calibration.schema_version == 2 else CASES


@dataclass(frozen=True)
class GroupedCalibrationConfig:
    assume_gaussian_independent_groups: bool
    assume_noninformative_counts: bool
    model: IntegrativeConfig
    n_simulations: int = 99
    alpha: float = 0.05
    random_seed: int = 20261024
    rank_tol: float = 1e-8
    gradient_tol: float = 1e-5
    max_iterations: int = 1000
    max_archive_bytes: int = 2147483648
    schema_version: int = 1
    solver: BoundarySolverConfig | None = None

    def __post_init__(self):
        if (
            self.assume_gaussian_independent_groups is not True
            or self.assume_noninformative_counts is not True
            or not isinstance(self.model, IntegrativeConfig)
        ):
            raise ValueError(
                "Explicitly assume independent Gaussian groups and noninformative counts"
            )
        if any(v.pca_fit == "target_means" for v in self.model.views):
            if self.schema_version != 2:
                raise ValueError("Target-mean PCA significance calibration requires schema 2")
        else:
            require_calibrated_pca(self.model)
        if self.model.covariance_type == "tied" and self.schema_version == 1:
            raise ValueError(
                "Shared-covariance significance calibration is not yet supported by legacy "
                "schema 1; use the boundary-capable schema 2"
            )
        for name in ("n_simulations", "max_iterations", "max_archive_bytes", "schema_version"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.schema_version not in {1, 2}
            or type(self.random_seed) is not int
            or not 0 <= self.random_seed < 2**64
        ):
            raise ValueError("Use schema 1/2 and a uint64 random seed")
        if (self.schema_version == 1 and self.solver is not None) or (
            self.schema_version == 2 and not isinstance(self.solver, BoundarySolverConfig)
        ):
            raise ValueError(
                "Schema 1 uses the legacy estimator; schema 2 requires a solver policy"
            )
        for name in ("alpha", "rank_tol", "gradient_tol"):
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
            or m.target_level not in {"population", "locality"}
            or m.missing_views != "error"
            or m.n_bootstrap != 0
            or m.max_k < 2
            or any(
                v.aggregation != ("mean" if v.kind == "morphological" else "none")
                or v.missing_policy != "error"
                or v.count_transform != "identity"
                for v in m.views
            )
        ):
            raise ValueError(
                "Use specimen morphology means and directly measured group ecology; "
                "no missingness or stability bootstrap; K>=2"
            )

    @classmethod
    def from_dict(cls, value):
        options = _options(
            cls,
            value,
            {
                "schema_version",
                "model",
                "assume_gaussian_independent_groups",
                "assume_noninformative_counts",
            },
        )
        if value.get("schema_version") == 1 and "solver" in value:
            raise ValueError("Legacy schema 1 does not contain a solver field")
        if "solver" in options:
            options["solver"] = BoundarySolverConfig.from_dict(options["solver"])
        options["model"] = IntegrativeConfig.from_dict(options["model"])
        return cls(**options)

    def to_dict(self):
        result = configuration_dict(self)
        if self.schema_version == 1:
            result.pop("solver")
        return result


@dataclass(frozen=True)
class GroupedStudyConfig:
    calibration: GroupedCalibrationConfig
    n_datasets: int = 25
    n_units: int = 64
    random_seed: int = 20261024
    schema_version: int = 1

    def __post_init__(self):
        if not isinstance(self.calibration, GroupedCalibrationConfig):
            raise ValueError("A grouped calibration policy is required")
        for name in ("n_datasets", "n_units", "schema_version"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.schema_version != 1
            or self.n_units < 20
            or self.n_units % 4
            or type(self.random_seed) is not int
            or not 0 <= self.random_seed < 2**64
        ):
            raise ValueError("Use schema 1, >=20 groups divisible by four and a uint64 seed")
        if 512 * self.n_units * 64 > self.calibration.model.max_working_bytes:
            raise ValueError("Study dimensions exceed workspace budget")

    @classmethod
    def from_dict(cls, value):
        options = _options(cls, value, {"schema_version", "calibration"})
        options["calibration"] = GroupedCalibrationConfig.from_dict(options["calibration"])
        return cls(**options)

    def to_dict(self):
        return {**configuration_dict(self), "calibration": self.calibration.to_dict()}
