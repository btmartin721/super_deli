"""Read the same sealed bytes that were verified, without model or array imports."""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import math
import re
import struct
from pathlib import Path, PurePosixPath

from super_deli.io.config import decode_json
from super_deli.io.delimited import identifier


class ArchiveError(ValueError):
    """A missing, unsupported, or inconsistent matched-multimodal artifact."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ArchiveError(message)


def id_index(records, key, context):
    """Bind unique biological or artifact IDs to their records."""
    result = {}
    for record in records:
        value = record[key]
        require(isinstance(value, str), f"{context}: invalid {key}")
        identifier(value, context)
        require(value not in result, f"{context}: repeated {key} {value!r}")
        result[value] = record
    return result


class SealedRun:
    def __init__(self, run_dir: Path, study_manifest: Path):
        self.run = run_dir.resolve(strict=True)
        self.manifest = study_manifest.resolve(strict=True)
        self.root = self.manifest.parent
        require(
            self.run.is_dir() and self.run.is_relative_to(self.root),
            "Run must be a directory under the supplied study manifest",
        )
        require(self.manifest.is_file(), "Study manifest must be a regular file")
        payload = self.manifest.read_bytes()
        self.manifest_sha256 = hashlib.sha256(payload).hexdigest()
        self.study_entries = decode_json(payload)
        require(isinstance(self.study_entries, dict), "Unsupported study manifest schema")
        self.native_digests: dict | None = None
        self.verified: dict[str, dict] = {}

    def read(self, name: str, *, optional: bool = False) -> bytes | None:
        relative = PurePosixPath(name)
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"Invalid archive artifact path: {name}",
        )
        path = self.run / relative
        require(path.resolve().is_relative_to(self.run), f"Artifact escapes run: {name}")
        key = path.relative_to(self.root).as_posix()
        if optional and key not in self.study_entries and not (path.exists() or path.is_symlink()):
            require(
                self.native_digests is None or name not in self.native_digests,
                f"Missing artifact declared by the run: {name}",
            )
            return None
        entry = self.study_entries.get(key)
        require(
            isinstance(entry, dict)
            and type(entry.get("bytes")) is int
            and entry["bytes"] >= 0
            and isinstance(entry.get("sha256"), str)
            and re.fullmatch(r"[a-f0-9]{64}", entry["sha256"]) is not None,
            f"Missing or invalid study manifest entry for {key}",
        )
        require(path.is_file(), f"Missing sealed artifact: {key}")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        require(
            len(data) == entry["bytes"] and digest == entry["sha256"],
            f"Digest or size mismatch for {key}",
        )
        if self.native_digests is not None:
            require(
                self.native_digests.get(name) == digest, f"Native run digest mismatch for {name}"
            )
        self.verified[name] = {"study_path": key, "sha256": digest, "bytes": len(data)}
        return data

    def json(self, name: str, *, optional: bool = False):
        data = self.read(name, optional=optional)
        if data is None:
            return None
        value = decode_json(data)
        require(value is not None, f"{name}: expected a JSON record or array")
        return value

    def csv(self, name: str, columns: set[str]) -> dict:
        return assignment_rows(self.read(name), columns, name)


def assignment_rows(data: bytes, columns: set[str], name: str) -> dict:
    """Read ID-indexed CSV rows from the bytes used for provenance."""
    try:
        with io.StringIO(data.decode("utf-8-sig"), newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            header = reader.fieldnames or []
            require(
                len(header) == len(columns) and set(header) == columns,
                f"{name}: unexpected CSV columns",
            )
            rows = list(reader)
    except (csv.Error, UnicodeError) as exc:
        raise ArchiveError(f"{name}: invalid UTF-8 CSV: {exc}") from exc
    require(
        all(None not in row and None not in row.values() for row in rows),
        f"{name}: malformed CSV row",
    )
    return id_index(rows, "unit_id", name)


def _npy_payload(data: bytes, shape: tuple[int, ...], formats: dict[str, str], name: str):
    """Decode the local v1.0 array header and validate the exact payload length."""
    message = f"Unsupported or malformed genomic {name}"
    require(data.startswith(b"\x93NUMPY\x01\x00") and len(data) >= 10, message)
    header_end = 10 + int.from_bytes(data[8:10], "little")
    try:
        header = ast.literal_eval(data[10:header_end].decode("ascii").strip())
    except (SyntaxError, UnicodeError, ValueError, RecursionError) as exc:
        raise ArchiveError(message) from exc
    require(
        isinstance(header, dict)
        and isinstance(header.get("descr"), str)
        and header["descr"][1:] in formats
        and header["descr"][0] in "<>"
        and type(header.get("fortran_order")) is bool
        and header.get("shape") == shape,
        message,
    )
    descr = header["descr"]
    format_code = formats[descr[1:]]
    elements = math.prod(shape)
    require(header_end + elements * struct.calcsize(descr[0] + format_code) == len(data), message)
    return header["fortran_order"], descr[0], format_code, memoryview(data)[header_end:]


def read_genomic_labels(data: bytes, rows: int) -> tuple[int, ...]:
    _, endian, code, values = _npy_payload(data, (rows,), {"i4": "i", "i8": "q"}, "labels.npy")
    return tuple(label for (label,) in struct.iter_unpack(endian + code, values))


def read_membership_matrix(data: bytes, rows: int, columns: int) -> tuple[tuple[float, ...], ...]:
    """Decode the native NumPy v1.0 float64 membership artifact."""
    fortran, endian, code, values = _npy_payload(
        data, (rows, columns), {"f8": "d"}, "membership.npy"
    )
    width = rows if fortran else columns
    vectors = struct.iter_unpack(f"{endian}{width}{code}", values)
    return tuple(zip(*vectors, strict=True)) if fortran else tuple(vectors)
