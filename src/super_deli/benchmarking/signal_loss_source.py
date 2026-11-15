"""Fail-closed, checksum-bound reads from the completed multimodal study."""

import hashlib
import json
import stat
from pathlib import Path

from super_deli.benchmarking.signal_loss_config import SOURCE_HEAD, SOURCE_VERSION


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class SealedStudy:
    source_head = SOURCE_HEAD
    source_version = SOURCE_VERSION

    def __init__(self, root, config):
        self.root = Path(root).resolve(strict=True)
        self.config = config
        self.consumed = {}
        for name, expected in (
            ("completion.json", config.source_completion_sha256),
            ("study-manifest.json", config.source_manifest_sha256),
        ):
            p = self.root / name
            if (
                not p.is_file()
                or p.is_symlink()
                or p.stat().st_size > config.max_file_bytes
                or digest(p) != expected
            ):
                raise ValueError("Source seal/size mismatch: " + name)
        self.completion = json.loads((self.root / "completion.json").read_text())
        c = self.completion
        if (
            c.get("completed") is not True
            or (config.fixture and c.get("fixture") is not True)
            or (
                not config.fixture
                and (
                    c.get("head_sha") != self.source_head or c.get("version") != self.source_version
                )
            )
        ):
            raise ValueError(
                "Require the declared completed source, or an explicit invented fixture"
            )
        self.inventory = json.loads((self.root / "study-manifest.json").read_text())
        if not isinstance(self.inventory, dict):
            raise ValueError("Invalid study inventory")

    def path(self, name):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Source path must stay inside the study")
        key = relative.as_posix()
        record = self.inventory.get(key)
        p = self.root / relative
        if (
            not isinstance(record, dict)
            or set(record) != {"bytes", "sha256"}
            or not p.resolve(strict=True).is_relative_to(self.root)
            or any(q.is_symlink() for q in [p, *p.parents] if q.is_relative_to(self.root))
            or not stat.S_ISREG(p.lstat().st_mode)
            or p.stat().st_size != record["bytes"]
            or p.stat().st_size > self.config.max_file_bytes
            or digest(p) != record["sha256"]
        ):
            raise ValueError("Source artifact identity/size/hash mismatch: " + key)
        self.consumed[key] = record["sha256"]
        return p

    def json(self, name):
        return json.loads(self.path(name).read_text())

    def arrays(self, name):
        import numpy as np

        value = np.load(self.path(name), allow_pickle=False)
        if isinstance(value, np.ndarray):
            return value.copy()
        with value:
            return {k: value[k].copy() for k in value.files}

    def reverify(self):
        for name in list(self.consumed):
            self.path(name)
        if (
            digest(self.root / "completion.json") != self.config.source_completion_sha256
            or digest(self.root / "study-manifest.json") != self.config.source_manifest_sha256
        ):
            raise ValueError("Source seals changed during diagnosis")

    def entries(self):
        summary = self.json("validation/main/summary.json")
        n = 1 if self.config.fixture else 24
        if summary.get("observations") != n or summary.get("passed") is not True:
            raise ValueError("Require the complete declared observation roster")
        rows = []
        for index in range(n):
            law = summary["records"][5 * index]["law"]
            prefix = f"validation/main/{index:02d}-{law}"
            rows.append(
                dict(
                    index=index,
                    law=law,
                    n=64,
                    analysis=prefix + "-analysis",
                    source=prefix + "-source",
                )
            )
        return rows
