"""Independent, opt-in contract for observed-coordinate Gaussian mixtures."""

import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class ObservedGMMConfig:
    feature_names: tuple[str, ...]
    max_k: int = 4
    min_cluster_size: int = 5
    n_init: int = 10
    max_iter: int = 2000
    tol: float = 1e-7
    fixed_point_tol: float = 1e-5
    covariance_bound: float = 1e-4
    min_coobserved: int = 5
    seed: int = 80423
    max_working_bytes: int = 268435456
    schema_version: int = 1

    def __post_init__(self):
        if (
            not isinstance(self.feature_names, tuple)
            or not 2 <= len(self.feature_names) <= 8
            or any(not isinstance(x, str) or not x or x != x.strip() for x in self.feature_names)
            or len(set(self.feature_names)) != len(self.feature_names)
        ):
            raise ValueError("Select 2–8 unique, unpadded continuous feature names")
        for name, low, high in (
            ("max_k", 1, 4),
            ("min_cluster_size", 5, 1000000),
            ("n_init", 1, 100),
            ("max_iter", 1, 20000),
            ("min_coobserved", 5, 1000000),
            ("seed", 0, 2**32 - 1),
            ("max_working_bytes", 1, 2**50),
            ("schema_version", 1, 1),
        ):
            v = getattr(self, name)
            if type(v) is not int or not low <= v <= high:
                raise ValueError(f"Invalid {name}")
        for name in ("tol", "fixed_point_tol", "covariance_bound"):
            v = getattr(self, name)
            if type(v) not in (int, float) or not math.isfinite(v) or not 0 < v < 1:
                raise ValueError(f"Invalid {name}")

    def to_dict(self):
        return {**asdict(self), "feature_names": list(self.feature_names)}

    @classmethod
    def from_dict(cls, value):
        if (
            not isinstance(value, dict)
            or "feature_names" not in value
            or set(value) - {f.name for f in fields(cls)}
            or not isinstance(value["feature_names"], list)
        ):
            raise ValueError("Expected a versioned observed-GMM configuration")
        return cls(**{**value, "feature_names": tuple(value["feature_names"])})
