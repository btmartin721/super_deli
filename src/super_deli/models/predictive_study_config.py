"""Explicit frozen predictive-screen study cells and independent seed domains."""

import math
from dataclasses import dataclass, replace

from super_deli.benchmarking.config import derived_seed
from super_deli.models.empirical_config import _options
from super_deli.models.integrative_config import IntegrativeConfig
from super_deli.models.predictive_config import PredictiveScreenConfig

LAWS = {
    "independent_gaussian",
    "correlated_gaussian",
    "zero_conditional_gaussian",
    "gaussian_scores",
    "complementary",
    "heavy_tailed",
    "cline",
    "unequal_variance",
}


@dataclass(frozen=True)
class PredictiveCell:
    name: str
    law: str
    role: str
    n_units: int
    n_datasets: int
    signal: float | None = None

    def __post_init__(self):
        if (
            not isinstance(self.name, str)
            or not self.name
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in self.name)
        ):
            raise ValueError("Safe unique cell names required")
        if not isinstance(self.law, str) or self.law not in LAWS:
            raise ValueError("Unknown predictive-validation law")
        role = (
            "alternative"
            if self.law == "complementary"
            else "stress"
            if self.law in {"heavy_tailed", "cline", "unequal_variance"}
            else "control"
        )
        if self.role != role:
            raise ValueError("Cell role disagrees with declared generating law")
        if type(self.n_units) is not int or self.n_units < 20 or self.n_units % 4:
            raise ValueError("Use >=20 units divisible by four")
        if type(self.n_datasets) is not int or self.n_datasets < 1:
            raise ValueError("Positive independent dataset count required")
        if self.law == "complementary":
            if (
                type(self.signal) not in (float, int)
                or not math.isfinite(self.signal)
                or self.signal <= 0
            ):
                raise ValueError("Positive finite generating signal required")
        elif self.signal is not None:
            raise ValueError("Only complementary alternatives take a signal multiplier")

    def to_dict(self):
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**_options(cls, value, {"name", "law", "role", "n_units", "n_datasets"}))


@dataclass(frozen=True)
class PredictiveStudyConfig:
    model: IntegrativeConfig
    screening: PredictiveScreenConfig
    cells: tuple[PredictiveCell, ...]
    n_test: int = 2048
    random_seed: int = 20261128
    max_artifact_bytes: int = 2 * 1024**3
    min_free_bytes: int = 8 * 1024**3
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported predictive-study schema")
        if not isinstance(self.screening, PredictiveScreenConfig):
            raise ValueError("Explicit screening configuration required")
        self.screening.validate_model(self.model)
        if (
            self.model.n_pcs != 4
            or self.model.max_k != 4
            or self.model.min_cluster_size != 5
            or self.model.target_level != "locality"
            or any(v.n_pcs != 2 or v.weight != 1 for v in self.model.views)
        ):
            raise ValueError("Use prespecified equal-weight four-PC grouped candidate search")
        if (
            not isinstance(self.cells, tuple)
            or not self.cells
            or any(not isinstance(c, PredictiveCell) for c in self.cells)
            or len({c.name for c in self.cells}) != len(self.cells)
        ):
            raise ValueError("Distinct explicit cells required")
        if self.model.covariance_type == "tied" and any(
            c.law == "gaussian_scores" for c in self.cells
        ):
            raise ValueError("Shared-family validation requires raw grouped views")
        if type(self.n_test) is not int or self.n_test < 20 or self.n_test % 4:
            raise ValueError("Use >=20 independent external units divisible by four")
        if self.n_test * 1024 > self.model.max_working_bytes:
            raise ValueError("External prediction exceeds the memory budget")
        if type(self.random_seed) is not int or not 0 <= self.random_seed < 2**32:
            raise ValueError("Unsigned 32-bit master seed required")
        if any(
            type(v) is not int or v < 1024**2
            for v in (self.max_artifact_bytes, self.min_free_bytes)
        ):
            raise ValueError("Explicit storage budgets >=1 MiB required")

    def to_dict(self):
        return {
            "schema_version": 1,
            "model": self.model.to_dict(),
            "screening": self.screening.to_dict(),
            "cells": [c.to_dict() for c in self.cells],
            "n_test": self.n_test,
            "random_seed": self.random_seed,
            "max_artifact_bytes": self.max_artifact_bytes,
            "min_free_bytes": self.min_free_bytes,
        }

    @classmethod
    def from_dict(cls, value):
        options = _options(cls, value, {"schema_version", "model", "screening", "cells"})
        options["model"] = IntegrativeConfig.from_dict(options["model"])
        options["screening"] = PredictiveScreenConfig.from_dict(options["screening"])
        if not isinstance(options["cells"], (list, tuple)):
            raise ValueError("cells must be an array")
        options["cells"] = tuple(PredictiveCell.from_dict(c) for c in options["cells"])
        return cls(**options)

    def plan(self):
        return [
            {
                "cell": c.name,
                "cell_config": c.to_dict(),
                "repetition": i,
                "data_seed": derived_seed(self.random_seed, c.name, i, "training_data"),
                "test_seed": derived_seed(self.random_seed, c.name, i, "external_data"),
                "split_seed": derived_seed(self.random_seed, c.name, i, "fold_split"),
                "planned_fits": 1 + self.screening.n_folds,
            }
            for c in self.cells
            for i in range(c.n_datasets)
        ]

    def probe(self, seed=20261129):
        return replace(
            self, cells=tuple(replace(c, n_datasets=1) for c in self.cells), random_seed=seed
        )
