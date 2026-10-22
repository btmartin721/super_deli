"""Explicit independent-block empirical genotype calibration declarations."""

import math
from dataclasses import asdict, dataclass

from super_deli.models.block_map import parse_blocks
from super_deli.models.config import BaselineConfig
from super_deli.models.pipeline_config import _object

PAIRED_CASES = ("common_hwe", "four_groups", "linked_stress", "cline_stress")
CASES = (*PAIRED_CASES, "moderate_ld", "split_block_stress")


@dataclass(frozen=True)
class LinkedCalibrationConfig:
    model: BaselineConfig
    block_ids: tuple[str, ...]
    assume_unrelated_specimens: bool
    assume_independent_blocks: bool
    assume_fixed_panel: bool
    assume_prespecified_blocks: bool
    modality: str = "genomic"
    n_simulations: int = 99
    alpha: float = 0.05
    random_seed: int = 20261027
    max_working_bytes: int = 268435456
    max_archive_bytes: int = 2147483648
    schema_version: int = 1

    def __post_init__(self):
        for name in (
            "assume_unrelated_specimens",
            "assume_independent_blocks",
            "assume_fixed_panel",
            "assume_prespecified_blocks",
        ):
            if getattr(self, name) is not True:
                raise ValueError(f"Explicitly declare {name}")
        if type(self.block_ids) is not tuple:
            raise ValueError("block_ids must be an immutable tuple")
        parse_blocks({"schema_version": 1, "block_ids": list(self.block_ids)})
        if len(set(self.block_ids)) < 2:
            raise ValueError("At least two declared independent blocks are required")
        if not isinstance(self.model, BaselineConfig):
            raise ValueError("A genomic baseline model is required")
        m = self.model
        if (
            m.assume_unlinked
            or m.n_bootstrap
            or m.max_k < 2
            or m.scaling != "patterson"
            or m.missing_policy != "error"
        ):
            raise ValueError(
                "Use linked complete data, Patterson scaling, K>=2 and no stability bootstrap"
            )
        if (
            not isinstance(self.modality, str)
            or not self.modality
            or self.modality != self.modality.strip()
        ):
            raise ValueError("modality must be nonempty and unpadded")
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

    def to_dict(self):
        return {**asdict(self), "block_ids": list(self.block_ids)}

    @classmethod
    def from_dict(cls, value):
        o = _object(
            cls,
            value,
            {
                "schema_version",
                "model",
                "block_ids",
                "assume_unrelated_specimens",
                "assume_independent_blocks",
                "assume_fixed_panel",
                "assume_prespecified_blocks",
            },
        )
        if not isinstance(o["block_ids"], list):
            raise ValueError("JSON block_ids must be an array")
        o["block_ids"] = parse_blocks({"schema_version": 1, "block_ids": o["block_ids"]})
        o["model"] = BaselineConfig.from_dict(o["model"])
        return cls(**o)


@dataclass(frozen=True)
class LinkedStudyConfig:
    calibration: LinkedCalibrationConfig
    n_units: int = 64
    n_datasets: int = 25
    random_seed: int = 20261026
    schema_version: int = 1

    def __post_init__(self):
        if not isinstance(self.calibration, LinkedCalibrationConfig):
            raise ValueError("A linked genomic calibration declaration is required")
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
        if self.calibration.modality != "genomic" or self.calibration.block_ids != study_blocks():
            raise ValueError("Study law requires genomic modality and sixteen eight-SNP blocks")
        if self.n_units < 2 * self.calibration.model.min_cluster_size:
            raise ValueError("Too few specimens for feasible K>=2")

    def to_dict(self):
        return {**asdict(self), "calibration": self.calibration.to_dict()}

    @classmethod
    def from_dict(cls, value):
        o = _object(cls, value, {"schema_version", "calibration"})
        o["calibration"] = LinkedCalibrationConfig.from_dict(o["calibration"])
        return cls(**o)


def study_blocks():
    return tuple(f"block_{j // 8:03d}" for j in range(128))
