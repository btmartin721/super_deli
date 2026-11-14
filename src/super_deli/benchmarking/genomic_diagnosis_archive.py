"""Checksum-bound, fail-closed reads from one completed archived study."""

import hashlib
import json
import stat
from pathlib import Path

import numpy as np

from super_deli.benchmarking.genomic_diagnosis_math import require


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class Archive:
    def __init__(self, directory, config):
        self.root = Path(directory).resolve(strict=True)
        self.config = config
        self.consumed = {}
        for name, expected in (
            ("completion.json", config.source_completion_sha256),
            ("verification-manifest.json", config.source_manifest_sha256),
        ):
            path = self.root / name
            require(path.is_file() and not path.is_symlink(), "Regular source receipt required")
            require(
                path.stat().st_size <= config.max_file_bytes and digest(path) == expected,
                "Source receipt checksum mismatch",
            )
        self.receipt = json.loads((self.root / "completion.json").read_text())
        require(
            self.receipt.get("completed") is True
            and self.receipt.get("head_sha") == config.source_commit
            and self.receipt.get("version") == config.source_version,
            "Source completion/version mismatch",
        )
        self.inventory = json.loads((self.root / "verification-manifest.json").read_text())
        self.base = self.root / "verification-evidence"

    def path(self, relative):
        relative = Path(relative)
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "Source path must remain inside the archive",
        )
        key = relative.as_posix()
        record = self.inventory.get(key)
        require(
            isinstance(record, dict) and "sha256" in record, "Missing manifested source artifact"
        )
        path = self.base / relative
        require(
            path.resolve(strict=True).is_relative_to(self.base.resolve()), "Escaping source path"
        )
        require(
            stat.S_ISREG(path.lstat().st_mode)
            and not any(
                parent.is_symlink() for parent in path.parents if parent != self.root.parent
            ),
            "Regular non-symlink source artifact required",
        )
        require(
            path.stat().st_size == record["bytes"]
            and path.stat().st_size <= self.config.max_file_bytes,
            "Source file size guard",
        )
        require(digest(path) == record["sha256"], "Consumed source checksum mismatch")
        self.consumed[key] = record["sha256"]
        return path

    def json(self, path):
        return json.loads(self.path(path).read_text())

    def arrays(self, path):
        with np.load(self.path(path), allow_pickle=False) as source:
            if isinstance(source, np.ndarray):
                return source.copy()
            return {key: source[key].copy() for key in source.files}

    def array(self, path):
        return np.load(self.path(path), allow_pickle=False)

    def reverify(self):
        for key in tuple(self.consumed):
            self.path(key)
        require(
            digest(self.root / "completion.json") == self.config.source_completion_sha256
            and digest(self.root / "verification-manifest.json")
            == self.config.source_manifest_sha256,
            "Source seal changed during diagnosis",
        )
