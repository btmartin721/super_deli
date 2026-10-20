"""Strict, dependency-free CSV/TSV structure shared by modality readers."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterator
from pathlib import Path


def identifier(value: str, location: str) -> str:
    if not value or value != value.strip() or "\x00" in value:
        raise ValueError(f"{location}: identifiers must be nonempty and unpadded without NUL")
    return value


def delimited_rows(
    path: Path,
    *,
    delimiter: str,
    required: set[str],
    context: str,
    optional: frozenset[str] = frozenset(),
) -> Iterator[tuple[int, dict[str, str]]]:
    """Yield rows with their ending physical line after checking the source header."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter, strict=True)
        try:
            header = next(reader, [])
            if not header:
                raise ValueError(f"{context}: missing table header")
            for column in header:
                identifier(column, f"{context} header")
            if duplicate := [name for name, n in Counter(header).items() if n > 1]:
                raise ValueError(f"{context}: duplicate columns {duplicate}")
            missing, extra = required - set(header), set(header) - required - optional
            if missing or extra:
                raise ValueError(
                    f"{context}: header mismatch: missing={sorted(missing)}; "
                    f"undeclared={sorted(extra)}"
                )
            for row in reader:
                if len(row) != len(header):
                    raise ValueError(
                        f"{context} line {reader.line_num}: row has {len(row)} fields, "
                        f"expected {len(header)}"
                    )
                yield reader.line_num, dict(zip(header, row, strict=True))
        except (csv.Error, UnicodeError) as exc:
            raise ValueError(
                f"{context} line {max(1, reader.line_num)}: invalid UTF-8 CSV/TSV: {exc}"
            ) from exc
