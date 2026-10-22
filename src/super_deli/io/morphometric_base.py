"""Shared fixed-landmark parsing rules; no geometric transformations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from super_deli.core.data import Modality, MorphometricReaderOptions
from super_deli.core.morphometrics import LandmarkRecord
from super_deli.io.delimited import identifier
from super_deli.io.numeric import finite_float


class MorphometricIngestionError(ValueError):
    """A source violates declared coordinate geometry, identities or supported syntax."""


def options_for(modality: Modality) -> MorphometricReaderOptions:
    if modality.morphometric_options is None:
        raise MorphometricIngestionError(
            f"{modality.name}: declare morphometric_options with dimensions and landmark_ids"
        )
    return modality.morphometric_options


def coordinates(
    tokens: Sequence[str], options: MorphometricReaderOptions, location: str
) -> tuple[float | None, ...]:
    if len(tokens) != options.dimensions:
        raise MorphometricIngestionError(
            f"{location}: expected {options.dimensions} coordinate fields"
        )
    point = tuple(
        None if text in options.missing_values else finite_float(text, location) for text in tokens
    )
    if options.missing_landmark and point == options.missing_landmark:
        return (None,) * options.dimensions
    return point


def scale_factor(text: str, location: str) -> float:
    number = finite_float(text, location)
    if number <= 0:
        raise MorphometricIngestionError(f"{location}: scale factor must be positive")
    return number


def unit_for(observation: str, links: dict[str, str], location: str) -> str:
    identifier(observation, location)
    if observation not in links:
        raise MorphometricIngestionError(f"{location}: undeclared observation ID {observation!r}")
    return links[observation]


def check_coverage(records: tuple[LandmarkRecord, ...], modality: Modality) -> None:
    ids = [r.observation_id for r in records]
    if duplicates := [name for name, count in Counter(ids).items() if count > 1]:
        raise MorphometricIngestionError(f"{modality.name}: duplicate observation IDs {duplicates}")
    if missing := {o.observation_id for o in modality.observations} - set(ids):
        raise MorphometricIngestionError(
            f"{modality.name}: missing declared observations {sorted(missing)}"
        )
