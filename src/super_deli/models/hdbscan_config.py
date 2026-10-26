"""Dependency-free contract for a separate density-clustering family."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class HDBSCANConfig:
    assume_independent_units: bool
    schema_version: int = 1
    model_family: str = "tabular_pca_hdbscan"
    feature_names: tuple[str, ...] = ()
    n_pcs: int = 2
    min_cluster_size: int = 5
    min_samples: int = 5
    alpha: float = 1.0
    cluster_selection_method: str = "eom"
    cluster_selection_epsilon: float = 0.0
    allow_single_cluster: bool = True
    n_subsamples: int = 0
    subsample_fraction: float = 0.8
    random_seed: int = 0
    rank_tol: float = 1e-12
    max_working_bytes: int = 268435456
    max_pairwise_bytes: int = 268435456

    def __post_init__(self):
        if type(self.assume_independent_units) is not bool or not self.assume_independent_units:
            raise ValueError("Explicitly assume independent biological units")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported HDBSCAN schema version")
        if self.model_family != "tabular_pca_hdbscan":
            raise ValueError("Unsupported model_family; no GMM calibration is available")
        for name in ("n_pcs", "min_samples", "max_working_bytes", "max_pairwise_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.min_cluster_size) is not int or self.min_cluster_size < 2:
            raise ValueError("min_cluster_size must be an integer at least two")
        for name in ("n_subsamples", "random_seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("alpha", "cluster_selection_epsilon", "subsample_fraction", "rank_tol"):
            v = getattr(self, name)
            if type(v) not in (int, float) or not math.isfinite(v):
                raise ValueError(f"{name} must be finite and numeric")
        if self.alpha <= 0 or self.cluster_selection_epsilon < 0:
            raise ValueError("alpha must be positive and epsilon nonnegative")
        if not 0 < self.subsample_fraction < 1 or not 0 < self.rank_tol < 1:
            raise ValueError("subsample_fraction and rank_tol must be between zero and one")
        if self.cluster_selection_method not in ("eom", "leaf"):
            raise ValueError("cluster_selection_method must be eom or leaf")
        if type(self.allow_single_cluster) is not bool:
            raise ValueError("allow_single_cluster must be boolean")
        if (
            not isinstance(self.feature_names, tuple)
            or any(not isinstance(n, str) or not n or n != n.strip() for n in self.feature_names)
            or len(set(self.feature_names)) != len(self.feature_names)
        ):
            raise ValueError("feature_names must contain unique nonempty, unpadded names")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("HDBSCAN configuration must be an object")
        required = {"schema_version", "model_family", "assume_independent_units"}
        if required - value.keys() or value.keys() - {f.name for f in fields(cls)}:
            raise ValueError("Required HDBSCAN fields are missing or unknown fields were supplied")
        result = dict(value)
        if "feature_names" in result:
            if not isinstance(result["feature_names"], (tuple, list)):
                raise ValueError("feature_names must be an array")
            result["feature_names"] = tuple(result["feature_names"])
        return cls(**result)

    def preprocessing_config(self):
        """Adapt shared preprocessing only; never dispatch this family to a GMM."""
        from super_deli.models.tabular_cluster_config import TabularClusterConfig

        return TabularClusterConfig(
            True,
            n_pcs=self.n_pcs,
            min_cluster_size=self.n_pcs + 1,
            n_bootstrap=0,
            feature_names=self.feature_names,
            missing_policy="error",
            scaling="standard",
            rank_tol=self.rank_tol,
            max_working_bytes=self.max_working_bytes,
        )
