"""Declared assumptions for a pooled, unlinked diploid genomic fitted null."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from super_deli.models.config import BaselineConfig
from super_deli.models.pipeline_config import _object

CASES = ("common_hwe", "rare_missing_hwe", "four_groups", "linked_stress", "cline_stress")


@dataclass(frozen=True)
class GenomicCalibrationConfig:
    model: BaselineConfig
    assume_unrelated_specimens: bool
    assume_hardy_weinberg: bool
    assume_fixed_panel: bool
    assume_noninformative_missingness: bool
    modality: str = "genomic"
    n_simulations: int = 99
    alpha: float = 0.05
    random_seed: int = 20261026
    max_working_bytes: int = 268435456
    max_archive_bytes: int = 2147483648
    schema_version: int = 1

    def __post_init__(self):
        for name in (
            "assume_unrelated_specimens",
            "assume_hardy_weinberg",
            "assume_fixed_panel",
            "assume_noninformative_missingness",
        ):
            if getattr(self, name) is not True:
                raise ValueError(f"Explicitly declare {name}")
        if not isinstance(self.model, BaselineConfig):
            raise ValueError("A diploid baseline model is required")
        if (
            not isinstance(self.modality, str)
            or not self.modality.strip()
            or self.modality != self.modality.strip()
        ):
            raise ValueError("modality must be a nonempty, unpadded name")
        for name in ("n_simulations", "max_working_bytes", "max_archive_bytes", "schema_version"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.schema_version != 1
            or type(self.random_seed) is not int
            or not 0 <= self.random_seed < 2**64
        ):
            raise ValueError("Use schema version 1 and a uint64 seed")
        if (
            type(self.alpha) not in (int, float)
            or not math.isfinite(self.alpha)
            or not 0 < self.alpha < 1
        ):
            raise ValueError("alpha must be finite and strictly between zero and one")
        m = self.model
        if not m.assume_unlinked or m.n_bootstrap or m.max_k < 2 or m.scaling != "patterson":
            raise ValueError(
                "Use explicitly unlinked loci, Patterson scaling, K>=2 and no stability bootstrap"
            )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        options = _object(
            cls,
            value,
            {
                "schema_version",
                "model",
                "assume_unrelated_specimens",
                "assume_hardy_weinberg",
                "assume_fixed_panel",
                "assume_noninformative_missingness",
            },
        )
        options["model"] = BaselineConfig.from_dict(options["model"])
        return cls(**options)


@dataclass(frozen=True)
class GenomicStudyConfig:
    calibration: GenomicCalibrationConfig
    n_units: int = 64
    n_datasets: int = 25
    random_seed: int = 20261026
    schema_version: int = 1

    def __post_init__(self):
        if not isinstance(self.calibration, GenomicCalibrationConfig):
            raise ValueError("A genomic calibration declaration is required")
        if type(self.n_units) is not int or self.n_units < 20 or self.n_units % 4:
            raise ValueError("Use >=20 specimens divisible by four")
        if (
            type(self.n_datasets) is not int
            or self.n_datasets < 1
            or type(self.schema_version) is not int
            or self.schema_version != 1
        ):
            raise ValueError("Use positive n_datasets and schema version 1")
        if type(self.random_seed) is not int or not 0 <= self.random_seed < 2**64:
            raise ValueError("Use a uint64 study seed")
        if self.calibration.model.missing_policy != "mean":
            raise ValueError("The rare/missing validation case requires mean imputation")
        if self.n_units < 2 * self.calibration.model.min_cluster_size:
            raise ValueError("Too few specimens for a feasible K>=2")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        options = _object(cls, value, {"schema_version", "calibration"})
        options["calibration"] = GenomicCalibrationConfig.from_dict(options["calibration"])
        return cls(**options)
