"""Lossless local evidence compaction with verified, safe scratch materialization."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile, TemporaryDirectory

from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def _safe_name(name):
    path = PurePosixPath(name)
    if (
        not name
        or not path.parts
        or "\x00" in name
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != name
    ):
        raise ValueError("Unsafe or noncanonical evidence archive member")
    return path


def _files(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise ValueError("Evidence may contain only regular files/directories")
        if path.is_file():
            name = path.relative_to(directory).as_posix()
            _safe_name(name)
            result[name] = {
                "sha256": fingerprint_file(path).sha256,
                "size_bytes": path.stat().st_size,
            }
    if not result:
        raise ValueError("Cannot compact empty evidence")
    return result


def _verify(path, manifest, *, destination=None, max_unpacked_bytes=2 * 1024**3):
    if sum(v["size_bytes"] for v in manifest.values()) > max_unpacked_bytes:
        raise ValueError("Evidence archive exceeds its unpacked scratch budget")
    seen = set()
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            relative = _safe_name(member.name)
            if not member.isfile() or member.name in seen or member.name not in manifest:
                raise ValueError("Unexpected, duplicate or nonregular evidence member")
            expected = manifest[member.name]
            if member.size != expected["size_bytes"]:
                raise ValueError("Evidence member size differs")
            digest = hashlib.sha256()
            handle = archive.extractfile(member)
            output = None
            try:
                if destination is not None:
                    target = destination.joinpath(*relative.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    output = target.open("xb")
                while block := handle.read(1024 * 1024):
                    digest.update(block)
                    if output is not None:
                        output.write(block)
            finally:
                handle.close()
                if output is not None:
                    output.close()
            if digest.hexdigest() != expected["sha256"]:
                raise ValueError("Evidence member checksum differs")
            seen.add(member.name)
    if seen != manifest.keys():
        raise ValueError("Evidence archive is missing declared files")


def compact_directory(source, archive):
    """Verify every archived byte; caller may then remove its redundant working copy."""
    source, archive = Path(source).resolve(), Path(archive).resolve()
    manifest_path = archive.with_name(archive.name + ".manifest.json")
    partial = archive.with_name(archive.name + ".partial")
    if archive.is_relative_to(source) or any(p.exists() for p in (archive, manifest_path, partial)):
        raise ValueError("Use a fresh archive outside the source evidence directory")
    manifest = _files(source)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("xb") as handle:
        with tarfile.open(fileobj=handle, mode="w:gz", compresslevel=6) as target:
            for name, entry in manifest.items():
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = entry["size_bytes"], 0o644, 0
                with (source / name).open("rb") as original:
                    target.addfile(info, original)
        handle.flush()
        os.fsync(handle.fileno())
    _verify(partial, manifest)
    partial.rename(archive)
    write_json({"schema_version": 1, "files": manifest}, manifest_path)
    with manifest_path.open("rb") as handle:
        os.fsync(handle.fileno())
    return {
        "archive": archive.name,
        "archive_sha256": fingerprint_file(archive).sha256,
        "manifest": manifest_path.name,
        "manifest_sha256": fingerprint_file(manifest_path).sha256,
        "size_bytes": archive.stat().st_size,
        "files": len(manifest),
        "unpacked_bytes": sum(v["size_bytes"] for v in manifest.values()),
    }


@contextmanager
def materialize(archive, record, *, scratch, min_free_bytes=0):
    """Read immutable archived evidence without changing archived absolute provenance."""
    archive, scratch = Path(archive).resolve(), Path(scratch).resolve()
    manifest_path = archive.with_name(archive.name + ".manifest.json")
    if (
        archive.name != Path(record["archive"]).name
        or manifest_path.name != Path(record["manifest"]).name
        or fingerprint_file(archive).sha256 != record["archive_sha256"]
        or fingerprint_file(manifest_path).sha256 != record["manifest_sha256"]
    ):
        raise ValueError("Observation archive or manifest fingerprint differs")
    declaration = read_json(manifest_path)
    if archive.stat().st_size != record["size_bytes"]:
        raise ValueError("Observation archive byte count differs")
    if declaration["schema_version"] != 1 or len(declaration["files"]) != record["files"]:
        raise ValueError("Observation manifest schema/count differs")
    if sum(v["size_bytes"] for v in declaration["files"].values()) != record["unpacked_bytes"]:
        raise ValueError("Observation unpacked byte count differs")
    scratch.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(scratch).free < min_free_bytes + record["unpacked_bytes"]:
        raise RuntimeError("Insufficient checked scratch space for archived evidence")
    with TemporaryDirectory(prefix="super-deli-evidence-", dir=scratch) as temporary:
        destination = Path(temporary)
        _verify(archive, declaration["files"], destination=destination)
        yield destination


def write_progress(value, path):
    """Atomically replace mutable progress; scientific evidence stays exclusive."""
    path = Path(path)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".progress-", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(handle.name, path)


def retained_bytes(root):
    """Count this checkpoint's retained files, excluding its transient workspaces."""
    root = Path(root)
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file()
        and not {".working", ".scratch"}.intersection(path.relative_to(root).parts)
    )
