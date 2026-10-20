"""Immutable declarations of biological units and modality observations.

Identifiers are case-sensitive and never normalized. An observation identifies
a source record; its unit_id identifies the biological unit at the modality's
declared level. Multiple observations may refer to one biological unit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

SCHEMA_VERSION = 1
Level = Literal["specimen", "population", "locality"]
Kind = Literal["genomic", "morphological", "ecological", "morphometric"]
FeatureType = Literal["continuous", "count", "binary", "ordinal", "categorical"]


@dataclass(frozen=True)
class Specimen:
    specimen_id: str
    population_id: str | None = None
    locality_id: str | None = None


@dataclass(frozen=True)
class Observation:
    observation_id: str
    unit_id: str


@dataclass(frozen=True)
class Feature:
    name: str
    dtype: FeatureType
    unit: str | None = None
    categories: tuple[str, ...] = ()
    role: Literal["feature", "covariate"] = "feature"


@dataclass(frozen=True)
class GenomicReaderOptions:
    """Explicit format options; arbitrary SNPio keyword arguments are not accepted."""

    has_popids: bool = False
    has_marker_names: bool = False
    allele_start_col: int | None = None
    allele_encoding: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Modality:
    name: str
    kind: Kind
    level: Level
    format: str
    path: str
    observations: tuple[Observation, ...]
    features: tuple[Feature, ...] = ()
    popmapfile: str | None = None
    reader_options: GenomicReaderOptions | None = None


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str
    modalities: tuple[Modality, ...]
    specimens: tuple[Specimen, ...] = ()
    populations: tuple[str, ...] = ()
    localities: tuple[str, ...] = ()
    random_seed: int = 0
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        """Return independent JSON-compatible values, including arrays as lists."""
        import json

        result = json.loads(json.dumps(asdict(self), allow_nan=False))
        for modality in result["modalities"]:
            options = modality["reader_options"]
            if options is None:
                del modality["reader_options"]
            else:
                options["allele_encoding"] = dict(options["allele_encoding"])
        return result
