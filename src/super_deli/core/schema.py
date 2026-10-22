"""Strict structural parsing of versioned dataset declarations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from super_deli.core.data import (
    SCHEMA_VERSION,
    DatasetConfig,
    Feature,
    Modality,
    Observation,
    Specimen,
)


class ConfigError(ValueError):
    """A malformed or unsupported dataset declaration."""


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ConfigError(f"{location}: expected a nonempty string without surrounding whitespace")
    return value


def _integer(value: Any, location: str) -> int:
    if type(value) is not int or value < 0:
        raise ConfigError(f"{location}: expected a nonnegative integer")
    return value


def _array(value: Any, location: str, parser: Callable) -> tuple:
    if not isinstance(value, list):
        raise ConfigError(f"{location}: expected an array")
    return tuple(parser(item, f"{location}[{i}]") for i, item in enumerate(value))


class _Object:
    """Consume fields once and reject misspellings or unsupported extensions."""

    def __init__(self, value: Any, location: str):
        if not isinstance(value, dict):
            raise ConfigError(f"{location}: expected an object")
        self.fields = dict(value)
        self.location = location

    def get(self, key: str, parser: Callable, *, default: Any = ...):
        if key not in self.fields:
            if default is ...:
                raise ConfigError(f"{self.location}.{key}: required field")
            return default
        return parser(self.fields.pop(key), f"{self.location}.{key}")

    def optional_string(self, key: str) -> str | None:
        return self.get(key, lambda v, p: None if v is None else _string(v, p), default=None)

    def array(self, key: str, parser: Callable, *, required: bool = False) -> tuple:
        if required:
            return self.get(key, lambda v, p: _array(v, p, parser))
        return self.get(key, lambda v, p: _array(v, p, parser), default=())

    def finish(self):
        if self.fields:
            raise ConfigError(f"{self.location}: unknown fields: {', '.join(sorted(self.fields))}")


def _specimen(value: Any, location: str) -> Specimen:
    obj = _Object(value, location)
    result = Specimen(
        obj.get("specimen_id", _string),
        obj.optional_string("population_id"),
        obj.optional_string("locality_id"),
    )
    obj.finish()
    return result


def _observation(value: Any, location: str) -> Observation:
    obj = _Object(value, location)
    result = Observation(obj.get("observation_id", _string), obj.get("unit_id", _string))
    obj.finish()
    return result


def _feature(value: Any, location: str) -> Feature:
    obj = _Object(value, location)
    result = Feature(
        name=obj.get("name", _string),
        dtype=obj.get("dtype", _string),
        unit=obj.optional_string("unit"),
        categories=obj.array("categories", _string),
        role=obj.get("role", _string, default="feature"),
    )
    obj.finish()
    return result


def _modality(value: Any, location: str) -> Modality:
    obj = _Object(value, location)
    result = Modality(
        name=obj.get("name", _string),
        kind=obj.get("kind", _string),
        level=obj.get("level", _string),
        format=obj.get("format", _string),
        path=obj.get("path", _string),
        observations=obj.array("observations", _observation, required=True),
        features=obj.array("features", _feature),
        popmapfile=obj.optional_string("popmapfile"),
    )
    obj.finish()
    return result


def config_from_dict(value: Any) -> DatasetConfig:
    """Parse field types; biological consistency is checked by validate_dataset."""
    obj = _Object(value, "dataset")
    version = obj.get("schema_version", _integer)
    if version != SCHEMA_VERSION:
        raise ConfigError(f"dataset.schema_version: unsupported version {version}")
    result = DatasetConfig(
        dataset_id=obj.get("dataset_id", _string),
        modalities=obj.array("modalities", _modality, required=True),
        specimens=obj.array("specimens", _specimen),
        populations=obj.array("populations", _string),
        localities=obj.array("localities", _string),
        random_seed=obj.get("random_seed", _integer, default=0),
        schema_version=version,
    )
    obj.finish()
    return result
