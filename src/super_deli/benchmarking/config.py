"""Dependency-free, versioned declarations for a bounded integrative study."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, fields
from typing import Protocol

from super_deli.models.integrative_config import IntegrativeConfig, configuration_dict


class SimulationDimensions(Protocol):
    n_units: int
    n_loci: int
    max_group_members: int


KINDS = ("genomic", "morphological", "ecological", "morphometric")


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value):
        raise ValueError("Names must be safe lowercase identifiers of at most 64 characters")


def _options(cls, value, required):
    if not isinstance(value, dict):
        raise ValueError(f"{cls.__name__} must be an object")
    if value.keys() - {f.name for f in fields(cls)} or required - value.keys():
        raise ValueError(f"Invalid {cls.__name__} fields; required={sorted(required)}")
    return dict(value)


@dataclass(frozen=True)
class Scenario:
    name: str
    pattern: str = "concordant"
    linkage: str = "matched"
    strength: float = 3.0
    cell_missing_rate: float = 0.0
    view_missing_rate: float = 0.0
    unequal_groups: bool = False

    def __post_init__(self):
        _identifier(self.name)
        if not isinstance(self.pattern, str) or self.pattern not in {
            "null",
            "concordant",
            "complementary",
            "conflicting",
        }:
            raise ValueError("Unsupported generating pattern")
        if not isinstance(self.linkage, str) or self.linkage not in {"matched", "grouped"}:
            raise ValueError("linkage must be matched or grouped")
        for name in ("strength", "cell_missing_rate", "view_missing_rate"):
            v = getattr(self, name)
            try:
                finite = type(v) in (int, float) and math.isfinite(float(v))
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(f"{name} must be finite and numeric")
        if not 0 <= self.strength <= 10 or (self.pattern == "null" and self.strength != 0):
            raise ValueError("strength must be in [0,10] and zero for null structure")
        if self.pattern != "null" and self.strength == 0:
            raise ValueError("Structured scenarios require positive strength")
        if not 0 <= self.cell_missing_rate < 1 or not 0 <= self.view_missing_rate < 1:
            raise ValueError("Missingness rates must be in [0,1)")
        if type(self.unequal_groups) is not bool or (
            self.unequal_groups and self.linkage != "grouped"
        ):
            raise ValueError("Unequal source counts require grouped linkage")

    @classmethod
    def from_dict(cls, value):
        return cls(**_options(cls, value, {"name"}))


@dataclass(frozen=True)
class Arm:
    name: str
    kinds: tuple[str, ...]

    def __post_init__(self):
        _identifier(self.name)
        if (
            not isinstance(self.kinds, tuple)
            or not 2 <= len(self.kinds) <= 4
            or any(not isinstance(k, str) or k not in KINDS for k in self.kinds)
            or len(set(self.kinds)) != len(self.kinds)
        ):
            raise ValueError("An arm requires two to four unique modality kinds")

    @classmethod
    def from_dict(cls, value):
        options = _options(cls, value, {"name", "kinds"})
        if not isinstance(options["kinds"], (list, tuple)):
            raise ValueError("Arm kinds must be an array")
        return cls(**{**options, "kinds": tuple(options["kinds"])})


@dataclass(frozen=True)
class BenchmarkConfig:
    scenarios: tuple[Scenario, ...]
    arms: tuple[Arm, ...]
    model: IntegrativeConfig
    schema_version: int = 1
    random_seed: int = 20261020
    n_datasets: int = 3
    n_units: int = 64
    n_loci: int = 64
    max_group_members: int = 4

    def __post_init__(self):
        for name in ("schema_version", "n_datasets", "n_units", "n_loci", "max_group_members"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.schema_version != 1:
            raise ValueError("Unsupported benchmark schema version")
        if type(self.random_seed) is not int or not 0 <= self.random_seed < 2**64:
            raise ValueError("random_seed must be a uint64 integer")
        if self.n_units < 20 or self.n_units % 4 or self.n_loci < 8 or self.n_loci % 2:
            raise ValueError("Use at least 20 units divisible by four and 8 even-numbered loci")
        if self.max_group_members < 2:
            raise ValueError("max_group_members must be at least two")
        for values, cls in ((self.scenarios, Scenario), (self.arms, Arm)):
            if (
                not isinstance(values, tuple)
                or not values
                or any(not isinstance(v, cls) for v in values)
                or len({v.name for v in values}) != len(values)
            ):
                raise ValueError(f"Require uniquely named {cls.__name__} declarations")
        if not isinstance(self.model, IntegrativeConfig):
            raise ValueError("model must be an IntegrativeConfig")
        model = self.model
        if (
            128 * self.n_units * self.max_group_members * (self.n_loci + 64)
            > model.max_working_bytes
            or 128 * self.n_units**2 > model.max_pairwise_bytes
        ):
            raise ValueError("Simulation or pairwise workspace exceeds the declared model budgets")
        if (
            {v.kind for v in model.views} != set(KINDS)
            or any(v.modality != v.kind or v.aggregation != "none" for v in model.views)
            or model.target_level != "specimen"
            or model.random_seed != 0
            or model.n_bootstrap < 1
            or model.min_cluster_size > self.n_units
        ):
            raise ValueError(
                "Template needs all four kind-named views, specimen level, no aggregation, "
                "seed 0 (replaced by paired derived seeds), and positive bootstrap count"
            )
        for arm in self.arms:
            if model.n_pcs > sum(v.n_pcs for v in model.views if v.kind in arm.kinds):
                raise ValueError("Arm has too few declared score dimensions for joint PCA")

    @classmethod
    def from_dict(cls, value):
        options = _options(cls, value, {"schema_version", "scenarios", "arms", "model"})
        for name, kind in (("scenarios", Scenario), ("arms", Arm)):
            if not isinstance(options[name], (list, tuple)):
                raise ValueError(f"{name} must be an array")
            options[name] = tuple(kind.from_dict(v) for v in options[name])
        options["model"] = IntegrativeConfig.from_dict(options["model"])
        return cls(**options)

    def to_dict(self):
        return configuration_dict(self)


def derived_seed(master: int, scenario: str, repetition: int, purpose: str) -> int:
    """Stable streams: adding/reordering scenarios, arms or repetitions changes no old seed."""
    payload = json.dumps([master, scenario, repetition, purpose], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
