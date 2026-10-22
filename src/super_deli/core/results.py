"""Result conventions for later estimators; no statistical estimates are computed here."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Literal

from super_deli.core.data import Level


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty string without surrounding whitespace")


def _finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and numeric")


@dataclass(frozen=True)
class Assignment:
    unit_id: str
    label: str | None
    reason: str | None = None

    def __post_init__(self):
        _text(self.unit_id, "unit_id")
        if self.label is None:
            _text(self.reason, "reason for an unresolved assignment")
        else:
            _text(self.label, "label")
        if self.reason is not None:
            _text(self.reason, "reason")


@dataclass(frozen=True)
class Evidence:
    """Keep different forms of support distinguishable and explicitly scoped.

    This record enforces reporting requirements. It cannot establish whether an
    estimator, null hypothesis, resampling scheme, or interval is statistically valid.
    """

    kind: Literal["bootstrap_frequency", "membership_probability", "confidence_interval", "p_value"]
    target: str
    status: Literal["available", "not_computed", "unavailable"]
    method: str | None = None
    value: float | None = None
    interval: tuple[float, float] | None = None
    confidence_level: float | None = None
    replicates: int | None = None
    null_hypothesis: str | None = None
    multiplicity: str | None = None
    reason: str | None = None

    def __post_init__(self):
        kinds = {"bootstrap_frequency", "membership_probability", "confidence_interval", "p_value"}
        if self.kind not in kinds:
            raise ValueError("Unknown evidence kind")
        _text(self.target, "target")
        if self.status not in {"available", "not_computed", "unavailable"}:
            raise ValueError("Unknown evidence status")
        for name in ("method", "null_hypothesis", "multiplicity", "reason"):
            if (value := getattr(self, name)) is not None:
                _text(value, name)
        if self.replicates is not None:
            if type(self.replicates) is not int or self.replicates <= 0:
                raise ValueError("replicates must be a positive integer")
        if self.status != "available":
            _text(self.reason, "reason for unavailable or uncomputed evidence")
            if any(v is not None for v in (self.value, self.interval, self.confidence_level)):
                raise ValueError("Unavailable or uncomputed evidence cannot contain estimates")
            return
        _text(self.method, "method")
        if self.kind == "confidence_interval":
            if self.value is not None or self.interval is None or len(self.interval) != 2:
                raise ValueError("A confidence interval requires two endpoints and no scalar value")
            for endpoint in self.interval:
                _finite(endpoint, "interval endpoint")
            if self.interval[0] > self.interval[1]:
                raise ValueError("Interval endpoints must be ordered")
            _finite(self.confidence_level, "confidence_level")
            if not 0 < self.confidence_level < 1:
                raise ValueError("confidence_level must be between zero and one")
        else:
            _finite(self.value, "value")
            if not 0 <= self.value <= 1 or self.interval is not None:
                raise ValueError("Probability/frequency estimates require one value in [0, 1]")
            if self.confidence_level is not None:
                raise ValueError("confidence_level belongs to confidence intervals")
        if self.kind == "bootstrap_frequency" and self.replicates is None:
            raise ValueError("Bootstrap frequency requires the number of replicates")
        if self.kind == "p_value":
            _text(self.null_hypothesis, "null_hypothesis")
            _text(self.multiplicity, "multiplicity treatment, including 'none' when unadjusted")


@dataclass(frozen=True)
class DelimitationResult:
    dataset_id: str
    model: str
    level: Level
    assignments: tuple[Assignment, ...]
    evidence: tuple[Evidence, ...]
    configuration_sha256: str
    schema_version: int = 1

    def __post_init__(self):
        _text(self.dataset_id, "dataset_id")
        _text(self.model, "model")
        _text(self.configuration_sha256, "configuration_sha256")
        if not re.fullmatch(r"[0-9a-f]{64}", self.configuration_sha256):
            raise ValueError("configuration_sha256 must identify the validated configuration")
        if self.level not in {"specimen", "population", "locality"}:
            raise ValueError("Unknown inference level")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported result schema version")
        if not self.assignments:
            raise ValueError("A result requires assignments, which may be unresolved")
        ids = [assignment.unit_id for assignment in self.assignments]
        if len(ids) != len(set(ids)):
            raise ValueError("A result cannot assign the same biological unit more than once")
        if not self.evidence:
            raise ValueError(
                "Declare evidence, including explicit not_computed/unavailable records"
            )

    def to_dict(self) -> dict:
        return asdict(self)
