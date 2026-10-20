"""Dependency-free declarations for a known raw-data reference population."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace

from super_deli.benchmarking.config import KINDS, Scenario
from super_deli.models.integrative_config import IntegrativeConfig

REFERENCE_LAW = "four_view_continuous_population_v1"
CASES = ("matched_null", "matched_complementary", "grouped_null", "grouped_complementary")


def _object(cls, value, required):
    if (
        not isinstance(value, dict)
        or value.keys() - {f.name for f in fields(cls)}
        or required - value.keys()
    ):
        raise ValueError(f"Invalid {cls.__name__}; required fields: {sorted(required)}")
    return dict(value)


@dataclass(frozen=True)
class ReferencePopulation:
    n_units: int = 64
    n_loci: int = 64
    max_group_members: int = 4
    linkage: str = "matched"
    unequal_groups: bool = False
    cell_missing_rate: float = 0.0
    view_missing_rate: float = 0.0
    law: str = REFERENCE_LAW
    schema_version: int = 1

    def __post_init__(self):
        for name in ("n_units", "n_loci", "max_group_members", "schema_version"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.schema_version != 1
            or self.law != REFERENCE_LAW
            or self.n_units < 20
            or self.n_units % 4
            or self.n_loci < 8
            or self.n_loci % 2
            or self.max_group_members < 2
        ):
            raise ValueError("Use the supported law, >=20 units divisible by 4 and >=8 even loci")
        self.scenario()

    def scenario(self, pattern="null"):
        if pattern not in {"null", "complementary"}:
            raise ValueError("Only null and complementary validation patterns are supported")
        return Scenario(
            "reference",
            pattern=pattern,
            strength=0.0 if pattern == "null" else 3.0,
            linkage=self.linkage,
            unequal_groups=self.unequal_groups,
            cell_missing_rate=self.cell_missing_rate,
            view_missing_rate=self.view_missing_rate,
        )

    @classmethod
    def from_dict(cls, value):
        return cls(**_object(cls, value, {"schema_version", "law"}))

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class PipelineCalibrationConfig:
    assume_declared_reference: bool
    reference: ReferencePopulation
    model: IntegrativeConfig
    n_simulations: int = 99
    alpha: float = 0.05
    random_seed: int = 20261022
    max_archive_bytes: int = 2147483648
    schema_version: int = 1

    def __post_init__(self):
        if self.assume_declared_reference is not True:
            raise ValueError("Explicitly assume the declared raw-data reference law")
        if not isinstance(self.reference, ReferencePopulation) or not isinstance(
            self.model, IntegrativeConfig
        ):
            raise ValueError("A reference population and integrative model are required")
        for name in ("n_simulations", "max_archive_bytes", "schema_version"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.schema_version != 1
            or type(self.random_seed) is not int
            or not 0 <= self.random_seed < 2**64
        ):
            raise ValueError("Use schema version 1 and a uint64 random seed")
        try:
            valid_alpha = (
                type(self.alpha) in (int, float)
                and math.isfinite(self.alpha)
                and 0 < self.alpha < 1
            )
        except OverflowError:
            valid_alpha = False
        if not valid_alpha:
            raise ValueError("alpha must be finite and strictly between zero and one")
        m, r = self.model, self.reference
        if (
            len(m.views) != 4
            or {v.kind for v in m.views} != set(KINDS)
            or any(v.modality != v.kind for v in m.views)
            or m.n_bootstrap != 0
            or m.max_k < 2
            or r.n_units < 2 * m.min_cluster_size
        ):
            raise ValueError(
                "Use all four kind-named views, no stability bootstrap and feasible K>=2"
            )
        grouped = r.linkage == "grouped"
        if m.target_level != ("locality" if grouped else "specimen") or any(
            v.aggregation != ("mean" if grouped and v.kind != "ecological" else "none")
            for v in m.views
        ):
            raise ValueError("Reference linkage and explicit model aggregation disagree")
        if r.view_missing_rate and m.missing_views != "complete_cases":
            raise ValueError("Whole-view missingness requires explicit complete_cases")
        if (
            128 * r.n_units * r.max_group_members * (r.n_loci + 64) > m.max_working_bytes
            or 128 * r.n_units**2 > m.max_pairwise_bytes
        ):
            raise ValueError("Reference dimensions exceed numerical workspace budgets")

    @classmethod
    def from_dict(cls, value):
        options = _object(
            cls, value, {"schema_version", "assume_declared_reference", "reference", "model"}
        )
        options["reference"] = ReferencePopulation.from_dict(options["reference"])
        options["model"] = IntegrativeConfig.from_dict(options["model"])
        return cls(**options)

    def to_dict(self):
        return asdict(self)

    def for_linkage(self, grouped):
        reference = replace(
            self.reference, linkage="grouped" if grouped else "matched", unequal_groups=grouped
        )
        model = replace(
            self.model,
            target_level="locality" if grouped else "specimen",
            views=tuple(
                replace(v, aggregation="mean" if grouped and v.kind != "ecological" else "none")
                for v in self.model.views
            ),
        )
        return replace(self, reference=reference, model=model)


@dataclass(frozen=True)
class PipelineStudyConfig:
    calibration: PipelineCalibrationConfig
    n_datasets: int = 25
    random_seed: int = 20261022
    schema_version: int = 1

    def __post_init__(self):
        if (
            not isinstance(self.calibration, PipelineCalibrationConfig)
            or type(self.n_datasets) is not int
            or self.n_datasets < 1
            or type(self.random_seed) is not int
            or not 0 <= self.random_seed < 2**64
            or type(self.schema_version) is not int
            or self.schema_version != 1
        ):
            raise ValueError("Invalid pipeline validation study declaration")
        if (
            self.calibration.reference.linkage != "matched"
            or self.calibration.reference.unequal_groups
        ):
            raise ValueError("Study template must start with the matched reference design")

    @classmethod
    def from_dict(cls, value):
        options = _object(cls, value, {"schema_version", "calibration"})
        options["calibration"] = PipelineCalibrationConfig.from_dict(options["calibration"])
        return cls(**options)

    def to_dict(self):
        return asdict(self)
