"""Prepare a provisional fixed-order TPS check without copying or modifying source data."""

import re
from pathlib import Path

from super_deli import load_config, validate_dataset, write_config
from super_deli.core.provenance import fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.io.config import write_json
from super_deli.io.delimited import identifier
from super_deli.io.tps import read_tps


def _finish(fields: dict, blocks: list, path: Path) -> None:
    if set(fields) != {"ID", "IMAGE", "count"}:
        raise ValueError(f"{path.name}: require LM, ID and IMAGE in each record")
    blocks.append(dict(fields))


def prepare(directory: Path, output: Path, *, assume_fixed_order: bool) -> None:
    if not assume_fixed_order:
        raise ValueError("Explicit --assume-fixed-order is required for this provisional check")
    modalities, units, report = [], set(), []
    for path in sorted(p for p in directory.iterdir() if p.suffix.lower() == ".tps"):
        fields, blocks = {}, []

        with path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                match = re.fullmatch(r"\s*(LM3?|ID|IMAGE)\s*=(.*)", line.rstrip("\r\n"), re.I)
                if not match:
                    continue
                key, value = match[1].upper(), match[2]
                if key in {"LM", "LM3"}:
                    if fields:
                        _finish(fields, blocks, path)
                    if key != "LM" or not re.fullmatch(r"[0-9]+", value.strip()):
                        raise ValueError("This preparation utility requires fixed 2D LM records")
                    fields = {"count": int(value.strip())}
                else:
                    if key in fields:
                        raise ValueError(f"{path.name}: duplicate {key}")
                    fields[key] = identifier(value, path.name)
        if fields:
            _finish(fields, blocks, path)
        counts = {b["count"] for b in blocks}
        if len(counts) != 1 or min(counts) < 3:
            raise ValueError(f"{path.name}: inconsistent/absent landmark counts")
        count = counts.pop()
        observations = [{"observation_id": b["ID"], "unit_id": b["IMAGE"]} for b in blocks]
        units.update(b["IMAGE"] for b in blocks)
        modality = {
            "name": path.stem,
            "kind": "morphometric",
            "level": "specimen",
            "format": "tps",
            "path": str(path.resolve()),
            "observations": observations,
            "morphometric_options": {
                "dimensions": 2,
                "landmark_ids": [f"point_{i + 1:02d}" for i in range(count)],
                "coordinate_unit": "source_coordinate_unit",
            },
        }
        modalities.append(modality)
        fingerprint = fingerprint_file(path)
        report.append(
            {
                "file": path.name,
                "sha256": fingerprint.sha256,
                "size_bytes": fingerprint.size_bytes,
                "record_count": len(blocks),
                "landmark_count": count,
            }
        )
    if not modalities:
        raise ValueError("No TPS files found")
    dataset = config_from_dict(
        {
            "schema_version": 1,
            "dataset_id": "provisional_downloaded_tps",
            "specimens": [{"specimen_id": u} for u in sorted(units)],
            "modalities": modalities,
        }
    )
    validation = validate_dataset(dataset, base_dir=directory)
    if not validation.valid:
        raise ValueError(str(validation.to_dict()))
    # The metadata pass above establishes declarations only. The package's strict
    # reader verifies every coordinate/directive rather than silently accepting it.
    for modality in dataset.modalities:
        records = read_tps(Path(modality.path), modality)
        if len(records) != len(modality.observations):
            raise ValueError("TPS records disagree with prepared declarations")
    output.mkdir(exist_ok=False)
    write_config(dataset, output / "dataset.json")
    assert load_config(output / "dataset.json") == dataset
    write_json({"schema_version": 1}, output / "shape.json")
    write_json(
        {
            "scope": "provisional fixed-order check; landmark/sliding protocol unverified",
            "scale_units": "unknown; SCALE metadata not applied",
            "inputs": report,
            "observation_sets": "separate modalities; shared images are repeated digitizations",
            "redistribution": "raw TPS/images remain in their source directory",
        },
        output / "preparation.json",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing TPS files")
    parser.add_argument("--output-dir", type=Path, required=True, help="New preparation directory")
    parser.add_argument("--assume-fixed-order", action="store_true")
    args = parser.parse_args()
    prepare(args.directory, args.output_dir, assume_fixed_order=args.assume_fixed_order)
