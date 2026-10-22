"""Typed observations that retain biological levels, feature schemas and missingness."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from super_deli.core.data import Feature, Kind, Level

TabularValue = float | int | str | None


@dataclass(frozen=True)
class TabularTable:
    """Rows follow source order; value columns follow the declared feature order.

    Category labels stay strings, including ordinal and binary labels. None is
    the only missing representation. Repeated unit IDs remain separate observations.
    """

    modality_name: str
    kind: Kind
    level: Level
    features: tuple[Feature, ...]
    observation_ids: tuple[str, ...]
    unit_ids: tuple[str, ...]
    values: tuple[tuple[TabularValue, ...], ...]

    @property
    def missing_mask(self) -> tuple[tuple[bool, ...], ...]:
        return tuple(tuple(value is None for value in row) for row in self.values)

    def to_dict(self) -> dict:
        return {"schema_version": 1, **asdict(self), "missing_mask": self.missing_mask}
