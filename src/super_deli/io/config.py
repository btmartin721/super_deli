"""JSON configuration I/O, with no implicit file mutation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from super_deli.core.data import DatasetConfig
from super_deli.core.schema import ConfigError, config_from_dict


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str):
    raise ConfigError(f"nonfinite JSON constant: {value}")


def load_config(path: str | Path) -> DatasetConfig:
    """Read JSON only. Resolve input paths against this file's parent when validating."""
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
    except (json.JSONDecodeError, UnicodeError, RecursionError) as exc:
        raise ConfigError(f"invalid JSON configuration: {exc}") from exc
    return config_from_dict(value)


def write_json(value: Any, path: str | Path) -> None:
    """Write one new UTF-8 JSON file; never overwrite an existing artifact."""
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(encoded)


def write_config(config: DatasetConfig, path: str | Path) -> None:
    """Write a structurally valid configuration to a new file."""
    value = config_from_dict(config.to_dict()).to_dict()
    write_json(value, path)
