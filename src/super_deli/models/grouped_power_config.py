"""Explicit cell design for the bounded grouped power/recovery benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass

from super_deli.benchmarking.config import derived_seed
from super_deli.models.empirical_config import _options
from super_deli.models.grouped_likelihood_config import GroupedLikelihoodConfig


@dataclass(frozen=True)
class PowerCell:
    name: str
    law: str
    n_units: int
    n_datasets: int
    signal: float | None = None

    def __post_init__(self):
        if (
            not isinstance(self.name, str)
            or not self.name
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in self.name)
        ):
            raise ValueError("Cell names require lowercase letters, digits or underscores")
        if not isinstance(self.law, str) or self.law not in {
            "complementary",
            "independent_gaussian",
            "heavy_tailed",
            "cline",
        }:
            raise ValueError("Unknown power/recovery generating law")
        if type(self.n_units) is not int or self.n_units < 20 or self.n_units % 4:
            raise ValueError("Use at least 20 localities, divisible by four")
        if type(self.n_datasets) is not int or self.n_datasets < 1:
            raise ValueError("Each cell requires a positive dataset count")
        if self.has_groups:
            if (
                type(self.signal) not in {int, float}
                or not math.isfinite(self.signal)
                or self.signal <= 0
            ):
                raise ValueError("Four-group cells require a finite positive signal multiplier")
        elif self.signal is not None:
            raise ValueError("Null/stress cells do not accept a group signal")

    @property
    def has_groups(self):
        return self.law == "complementary"

    def to_dict(self):
        return {
            "name": self.name,
            "law": self.law,
            "n_units": self.n_units,
            "n_datasets": self.n_datasets,
            "signal": self.signal,
        }

    @classmethod
    def from_dict(cls, value):
        return cls(**_options(cls, value, {"name", "law", "n_units", "n_datasets"}))


@dataclass(frozen=True)
class GroupedPowerStudyConfig:
    calibration: GroupedLikelihoodConfig
    cells: tuple[PowerCell, ...]
    random_seed: int = 20261120
    stability_first: int = 5
    stability_draws: int = 25
    max_artifact_bytes: int = 8 * 1024**3
    min_free_bytes: int = 8 * 1024**3
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValueError("Unsupported grouped power study schema")
        if not isinstance(self.calibration, GroupedLikelihoodConfig):
            raise ValueError("Use the existing grouped likelihood calibration policy")
        if self.calibration.model.n_bootstrap:
            raise ValueError("Significance and stability resampling must be separate")
        if (
            not isinstance(self.cells, tuple)
            or not self.cells
            or any(not isinstance(c, PowerCell) for c in self.cells)
        ):
            raise ValueError("Declare a nonempty tuple of valid power cells")
        if len({c.name for c in self.cells}) != len(self.cells):
            raise ValueError("Duplicate power cell names")
        if type(self.random_seed) is not int or not 0 <= self.random_seed < 2**32:
            raise ValueError("Use an unsigned 32-bit study seed")
        for value, name in [
            (self.stability_first, "stability_first"),
            (self.stability_draws, "stability_draws"),
        ]:
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.stability_first and not self.stability_draws:
            raise ValueError("Stability observations require positive resampling draws")
        if any(c.has_groups and self.stability_first > c.n_datasets for c in self.cells):
            raise ValueError("Stability subset exceeds a cell's declared dataset count")
        for value in (self.max_artifact_bytes, self.min_free_bytes):
            if type(value) is not int or value < 1024**2:
                raise ValueError("Use explicit positive byte budgets of at least one MiB")

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "calibration": self.calibration.to_dict(),
            "cells": [c.to_dict() for c in self.cells],
            "random_seed": self.random_seed,
            "stability_first": self.stability_first,
            "stability_draws": self.stability_draws,
            "max_artifact_bytes": self.max_artifact_bytes,
            "min_free_bytes": self.min_free_bytes,
        }

    @classmethod
    def from_dict(cls, value):
        options = _options(cls, value, {"schema_version", "calibration", "cells"})
        options["calibration"] = GroupedLikelihoodConfig.from_dict(options["calibration"])
        options["cells"] = tuple(PowerCell.from_dict(c) for c in options["cells"])
        return cls(**options)

    def plan(self):
        records = []
        for cell in self.cells:
            for i in range(cell.n_datasets):
                stability = cell.has_groups and i < self.stability_first
                records.append(
                    {
                        "cell": cell.name,
                        "repetition": i,
                        "data_seed": derived_seed(self.random_seed, cell.name, i, "observed_data"),
                        "null_stream_seed": derived_seed(
                            self.random_seed, cell.name, i, "null_stream"
                        ),
                        "stability": stability,
                        "planned_fits": 1
                        + self.calibration.n_simulations
                        + ((1 + self.stability_draws) if stability else 0),
                    }
                )
        return records


def default_cells():
    return (
        tuple(
            PowerCell(f"four_n{n}_s{str(s).replace('.', '_')}", "complementary", n, 50, s)
            for n in (64, 256)
            for s in (1.5, 3.0, 4.5)
        )
        + tuple(PowerCell(f"null_n{n}", "independent_gaussian", n, 25) for n in (64, 256))
        + (
            PowerCell("heavy_n64", "heavy_tailed", 64, 25),
            PowerCell("cline_n64", "cline", 64, 25),
        )
    )
