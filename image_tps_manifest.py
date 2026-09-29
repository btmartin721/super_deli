"""Build a read-only specimen manifest from a CSV, images, and optional TPS files.

The specimen CSV requires ``specimen_id,group,split,image_file``. Optional
columns are ``tps_file,scale_value,scale_unit``. File names are relative to
``images_dir`` or ``tps_dir``. The CSV is the source of expected specimens, so
missing images and repeated IDs remain visible as rows in the output.

Example::

    python image_tps_manifest.py --images-dir /data/images \
        --specimens-csv /data/specimens.csv --tps-dir /data/tps \
        --expected-landmarks 12 --output /data/manifest.csv

Only ``--output`` is written, and an existing output file is never replaced.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


MANIFEST_COLUMNS: Tuple[str, ...] = (
    "image_path",
    "specimen_id",
    "group",
    "split",
    "tps_path",
    "landmark_count",
    "scale_value",
    "scale_unit",
    "check_status",
    "check_details",
)
REQUIRED_CSV_COLUMNS = frozenset(("specimen_id", "group", "split", "image_file"))
IMAGE_SUFFIXES = frozenset((".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"))


@dataclass(frozen=True)
class TpsRecord:
    landmark_count: Optional[int]
    scale_value: Optional[float]
    image_name: Optional[str]
    issues: Tuple[str, ...]


def _relative_file(root: Path, file_name: str) -> Path:
    """Resolve a CSV file name without allowing it to leave its input root."""
    raw_path = Path(file_name)
    if raw_path.is_absolute():
        raise ValueError("absolute_path")
    candidate = (root / raw_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("path_outside_input_directory") from exc
    return candidate


def _positive_float(raw_value: str) -> float:
    value = float(raw_value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("value must be positive and finite")
    return value


def _read_tps(path: Path, expected_landmarks: Optional[int]) -> TpsRecord:
    """Read one two-dimensional LM block and optional IMAGE/SCALE fields."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    lm_positions = [i for i, line in enumerate(lines) if line.upper().startswith("LM=")]
    if len(lm_positions) != 1:
        issue = "missing_lm_block" if not lm_positions else "multiple_lm_blocks"
        return TpsRecord(None, None, None, (issue,))

    issues: List[str] = []
    lm_position = lm_positions[0]
    try:
        declared_count = int(lines[lm_position].split("=", 1)[1].strip())
        if declared_count <= 0:
            raise ValueError
    except ValueError:
        declared_count = None
        issues.append("invalid_lm_count")

    coordinate_lines: List[str] = []
    for line in lines[lm_position + 1 :]:
        if "=" in line:
            break
        if line:
            coordinate_lines.append(line)
    for line in coordinate_lines:
        parts = line.split()
        try:
            if len(parts) != 2 or not all(math.isfinite(float(part)) for part in parts):
                raise ValueError
        except ValueError:
            issues.append("invalid_coordinates")
            break

    landmark_count = len(coordinate_lines)
    if declared_count is not None and landmark_count != declared_count:
        issues.append("tps_landmark_count_mismatch")
    if expected_landmarks is not None and landmark_count != expected_landmarks:
        issues.append("unexpected_landmark_count")

    image_values = [line.split("=", 1)[1].strip() for line in lines if line.upper().startswith("IMAGE=")]
    if len(image_values) > 1:
        issues.append("multiple_image_fields")
    image_name = Path(image_values[0]).name if image_values and image_values[0] else None

    scale_values = [line.split("=", 1)[1].strip() for line in lines if line.upper().startswith("SCALE=")]
    scale_value = None
    if len(scale_values) > 1:
        issues.append("multiple_scale_fields")
    elif scale_values:
        try:
            scale_value = _positive_float(scale_values[0])
        except ValueError:
            issues.append("invalid_tps_scale")
    return TpsRecord(landmark_count, scale_value, image_name, tuple(issues))


def build_manifest(
    images_dir: Path,
    specimens_csv: Path,
    tps_dir: Optional[Path] = None,
    expected_landmarks: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Return one checked manifest row for every row in the specimen CSV.

    This function reads source files but never creates, edits, or deletes them.
    TPS files must contain one two-dimensional ``LM=`` block per file.
    """
    images_dir = Path(images_dir).resolve()
    specimens_csv = Path(specimens_csv).resolve()
    tps_dir = Path(tps_dir).resolve() if tps_dir is not None else images_dir
    if not images_dir.is_dir():
        raise ValueError(f"Image directory does not exist: {images_dir}")
    if not tps_dir.is_dir():
        raise ValueError(f"TPS directory does not exist: {tps_dir}")
    if expected_landmarks is not None and expected_landmarks <= 0:
        raise ValueError("expected_landmarks must be positive")

    with specimens_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing_columns = REQUIRED_CSV_COLUMNS.difference(reader.fieldnames or ())
        if missing_columns:
            raise ValueError("Missing specimen CSV columns: " + ", ".join(sorted(missing_columns)))
        specimens = list(reader)

    ids = Counter((row.get("specimen_id") or "").strip().casefold() for row in specimens)
    rows: List[Dict[str, str]] = []
    for specimen in specimens:
        specimen_id = (specimen.get("specimen_id") or "").strip()
        group = (specimen.get("group") or "").strip()
        split = (specimen.get("split") or "").strip()
        image_file = (specimen.get("image_file") or "").strip()
        tps_file = (specimen.get("tps_file") or "").strip()
        scale_raw = (specimen.get("scale_value") or "").strip()
        scale_unit = (specimen.get("scale_unit") or "").strip()
        issues: List[str] = []
        row = dict.fromkeys(MANIFEST_COLUMNS, "")
        row.update(specimen_id=specimen_id, group=group, split=split, scale_unit=scale_unit)

        if not specimen_id:
            issues.append("missing_specimen_id")
        elif ids[specimen_id.casefold()] > 1:
            issues.append("duplicate_specimen_id")
        if not group:
            issues.append("missing_group")
        if not split:
            issues.append("missing_split")

        if not image_file:
            issues.append("missing_image_file")
        else:
            try:
                image_path = _relative_file(images_dir, image_file)
                row["image_path"] = str(image_path)
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    issues.append("unsupported_image_extension")
                if not image_path.is_file():
                    issues.append("missing_image")
            except ValueError as exc:
                issues.append(str(exc))

        csv_scale = None
        if scale_raw:
            try:
                csv_scale = _positive_float(scale_raw)
                row["scale_value"] = str(csv_scale)
            except ValueError:
                issues.append("invalid_csv_scale")

        if tps_file:
            try:
                tps_path = _relative_file(tps_dir, tps_file)
                row["tps_path"] = str(tps_path)
                if not tps_path.is_file():
                    issues.append("missing_tps")
                else:
                    tps = _read_tps(tps_path, expected_landmarks)
                    issues.extend(tps.issues)
                    if tps.landmark_count is not None:
                        row["landmark_count"] = str(tps.landmark_count)
                    if tps.image_name and image_file and tps.image_name != Path(image_file).name:
                        issues.append("tps_image_mismatch")
                    if tps.scale_value is not None:
                        row["scale_value"] = str(tps.scale_value)
                        if csv_scale is not None and not math.isclose(csv_scale, tps.scale_value):
                            issues.append("scale_mismatch")
            except (OSError, UnicodeError) as exc:
                issues.append("unreadable_tps")
            except ValueError as exc:
                issues.append(str(exc))

        row["check_status"] = "error" if issues else "ok"
        row["check_details"] = ";".join(dict.fromkeys(issues))
        rows.append(row)
    return rows


def write_manifest(rows: Sequence[Dict[str, str]], output: Path) -> None:
    """Write a new CSV manifest without replacing an existing file."""
    with Path(output).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--specimens-csv", required=True, type=Path)
    parser.add_argument("--tps-dir", type=Path, help="Defaults to --images-dir")
    parser.add_argument("--expected-landmarks", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        rows = build_manifest(
            args.images_dir,
            args.specimens_csv,
            tps_dir=args.tps_dir,
            expected_landmarks=args.expected_landmarks,
        )
        write_manifest(rows, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"manifest error: {exc}\n")
    error_count = sum(row["check_status"] == "error" for row in rows)
    print(f"Wrote {len(rows)} rows to {args.output} ({error_count} with errors)")
    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
