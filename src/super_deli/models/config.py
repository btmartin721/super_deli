"""Explicit assumptions and numerical settings for the genomic baseline."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class BaselineConfig:
    ploidy: int
    missing_policy: str
    assume_independent_specimens: bool
    n_pcs: int = 2
    max_k: int = 4
    min_cluster_size: int = 3
    scaling: str = "patterson"
    n_bootstrap: int = 100
    assume_unlinked: bool = False
    n_init: int = 10
    max_iter: int = 500
    tol: float = 0.001
    reg_covar: float = 0.000001
    assignment_threshold: float = 0.0
    random_seed: int = 0
    n_threads: int = 1
    max_pairwise_bytes: int = 268435456
    schema_version: int = 1

    def __post_init__(self):
        for name in (
            "ploidy",
            "n_pcs",
            "max_k",
            "min_cluster_size",
            "n_init",
            "max_iter",
            "n_threads",
            "max_pairwise_bytes",
            "schema_version",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("n_bootstrap", "random_seed"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("assume_unlinked", "assume_independent_specimens"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be Boolean")
        if self.schema_version != 1 or self.ploidy != 2:
            raise ValueError("This baseline requires schema version 1 and declared diploid data")
        if not self.assume_independent_specimens:
            raise ValueError(
                "Explicitly declare the assumption of independent biological specimens"
            )
        if self.min_cluster_size < 2:
            raise ValueError("min_cluster_size must be at least two")
        if not isinstance(self.missing_policy, str) or self.missing_policy not in {"error", "mean"}:
            raise ValueError("missing_policy must be error or mean")
        if not isinstance(self.scaling, str) or self.scaling not in {
            "center",
            "standard",
            "patterson",
        }:
            raise ValueError("scaling must be center, standard, or patterson")
        for name in ("tol", "reg_covar", "assignment_threshold"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite and numeric")
        if self.tol <= 0 or self.reg_covar <= 0:
            raise ValueError("tol and reg_covar must be positive")
        if not 0 <= self.assignment_threshold <= 1:
            raise ValueError("assignment_threshold must be in [0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> BaselineConfig:
        if not isinstance(value, dict):
            raise ValueError("Baseline configuration must be an object")
        required = {"schema_version", "ploidy", "missing_policy", "assume_independent_specimens"}
        unknown = set(value) - {field.name for field in fields(cls)}
        if unknown or required - set(value):
            raise ValueError(
                f"Unknown baseline fields: {sorted(unknown)}; "
                f"missing: {sorted(required - set(value))}"
            )
        return cls(**value)
