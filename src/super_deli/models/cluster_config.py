"""Shared dependency-free settings for PCA/GMM clustering of independent units."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class ClusterConfig(ABC):
    assume_independent_units: bool
    n_pcs: int = 2
    max_k: int = 4
    min_cluster_size: int = 5
    n_bootstrap: int = 100
    n_init: int = 10
    max_iter: int = 500
    tol: float = 0.001
    reg_covar: float = 1e-6
    assignment_threshold: float = 0.0
    random_seed: int = 0
    n_threads: int = 1
    max_pairwise_bytes: int = 268435456

    @property
    @abstractmethod
    def schema_version(self) -> int:
        """Concrete configs place the version field after their own settings.

        This preserves the existing shape configuration's positional signature.
        """

    def __post_init__(self):
        if type(self.assume_independent_units) is not bool or not self.assume_independent_units:
            raise ValueError("Explicitly assume independent biological units")
        for name in (
            "n_pcs",
            "max_k",
            "min_cluster_size",
            "n_init",
            "max_iter",
            "n_threads",
            "max_pairwise_bytes",
            "schema_version",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("n_bootstrap", "random_seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.schema_version != 1:
            raise ValueError("Unsupported clustering configuration version")
        if self.min_cluster_size <= self.n_pcs:
            raise ValueError("min_cluster_size must exceed n_pcs for full covariance mixtures")
        for name in ("tol", "reg_covar", "assignment_threshold"):
            v = getattr(self, name)
            if type(v) not in (int, float) or not math.isfinite(v):
                raise ValueError(f"{name} must be finite and numeric")
        if self.tol <= 0 or self.reg_covar <= 0 or not 0 <= self.assignment_threshold <= 1:
            raise ValueError("Positive tol/reg_covar and assignment_threshold in [0, 1] required")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> ClusterConfig:
        if not isinstance(value, dict):
            raise ValueError("Clustering configuration must be an object")
        required = {"schema_version", "assume_independent_units"}
        unknown = value.keys() - {f.name for f in fields(cls)}
        if unknown or required - value.keys():
            raise ValueError(f"Required={sorted(required)}; unknown={sorted(unknown)}")
        return cls(**value)
