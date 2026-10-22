"""Reproducibility records for declarations and optionally their referenced files."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from super_deli import __version__
from super_deli.core.data import DatasetConfig
from super_deli.core.validation import input_paths


@dataclass(frozen=True)
class InputFingerprint:
    path: str
    sha256: str
    size_bytes: int


def fingerprint_file(path: str | Path) -> InputFingerprint:
    """Hash in bounded memory and reject a file that visibly changes during reading."""
    source = Path(path).resolve()
    if not source.is_file():
        raise OSError(f"Not a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise OSError(f"Not a regular file: {source}")
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    current = source.stat()

    def signature(info: os.stat_result) -> tuple:
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    if signature(before) != signature(after) or signature(after) != signature(current):
        raise OSError(f"Input changed while hashing: {source}")
    return InputFingerprint(str(source), digest.hexdigest(), after.st_size)


def configuration_hash(configuration: dict) -> str:
    """Identify a parsed configuration using canonical UTF-8 JSON."""
    canonical = json.dumps(
        configuration, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def collect_provenance(
    config: DatasetConfig, *, base_dir: str | Path, hash_inputs: bool = False
) -> dict:
    """Record the exact parsed configuration; hashing large inputs is opt-in.

    The configuration digest covers canonical UTF-8 JSON, not original whitespace.
    Referenced image-manifest contents are not traversed at this checkpoint.
    """
    configuration = config.to_dict()
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "configuration": configuration,
        "configuration_sha256": configuration_hash(configuration),
        "base_dir": str(Path(base_dir).resolve()),
        "random_seed": config.random_seed,
        "software": {"super_deli": __version__, "python": platform.python_version()},
        "platform": platform.platform(),
        "input_hash_status": "computed" if hash_inputs else "not_requested",
        "inputs": (
            [asdict(fingerprint_file(path)) for path in input_paths(config, base_dir)]
            if hash_inputs
            else []
        ),
    }
