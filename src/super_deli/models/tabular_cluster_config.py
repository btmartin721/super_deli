"""Explicit feature selection and preprocessing for numeric tabular clustering."""

from __future__ import annotations

import math
from dataclasses import dataclass

from super_deli.models.cluster_config import ClusterConfig


@dataclass(frozen=True)
class TabularClusterConfig(ClusterConfig):
    feature_names: tuple[str, ...] = ()
    missing_policy: str = "error"
    scaling: str = "standard"
    count_transform: str = "identity"
    rank_tol: float = 1e-12
    max_working_bytes: int = 268435456
    schema_version: int = 1

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.feature_names, tuple) or any(
            not isinstance(n, str) or not n or n != n.strip() for n in self.feature_names
        ):
            raise ValueError("feature_names must be a tuple of nonempty, unpadded names")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Duplicate selected feature names")
        for name, choices in (
            ("missing_policy", {"error", "mean"}),
            ("scaling", {"standard", "center"}),
            ("count_transform", {"identity", "log1p"}),
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or value not in choices:
                raise ValueError(f"{name} must be one of {sorted(choices)}")
        if (
            type(self.rank_tol) not in (int, float)
            or not math.isfinite(self.rank_tol)
            or not 0 < self.rank_tol < 1
        ):
            raise ValueError("rank_tol must be finite and between zero and one")
        if type(self.max_working_bytes) is not int or self.max_working_bytes < 1:
            raise ValueError("max_working_bytes must be a positive integer")

    @classmethod
    def from_dict(cls, value: object) -> TabularClusterConfig:
        if isinstance(value, dict) and "feature_names" in value:
            names = value["feature_names"]
            if not isinstance(names, (tuple, list)):
                raise ValueError("feature_names must be an array")
            value = {**value, "feature_names": tuple(names)}
        return super().from_dict(value)
