"""Stage isolated, fingerprinted copies of modality input files."""

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path

from super_deli.core.provenance import fingerprint_file


def stage_input(source: Path, directory: Path) -> tuple[Path, dict]:
    """Copy into a new directory and verify byte identity against the source."""
    directory.mkdir()
    staged = directory / source.name
    shutil.copyfile(source, staged)
    original = fingerprint_file(source)
    snapshot = fingerprint_file(staged)
    if original.sha256 != snapshot.sha256:
        raise OSError(f"Input changed while staging: {source}")
    return staged, asdict(original)
