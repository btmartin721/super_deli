"""Read-only paired comparison of the identical four original observed cohorts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json
from super_deli.models.grouped_config import CASES, GroupedStudyConfig


def compare_grouped_studies(legacy_root, current_root, *, output_dir):
    legacy_root, current_root, destination = map(
        lambda p: Path(p).resolve(), (legacy_root, current_root, output_dir)
    )
    if any(destination == p or destination.is_relative_to(p) for p in (legacy_root, current_root)):
        raise ValueError("Comparison must be outside both immutable study roots")
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    rows, hashes = [], {}
    for case in CASES:
        shards = [r / case for r in (legacy_root, current_root)]
        declarations = [GroupedStudyConfig.from_dict(read_json(r / "study.json")) for r in shards]
        old, new = declarations
        if (
            old.calibration.schema_version != 1
            or new.calibration.schema_version != 2
            or old.n_datasets != new.n_datasets
            or old.n_units != new.n_units
            or old.random_seed != new.random_seed
            or old.calibration.model != new.calibration.model
        ):
            raise ValueError(
                "Paired studies require identical observed design/seeds/model and schema 1 versus 2"
            )
        ledgers = []
        for shard in shards:
            path = shard / "datasets.jsonl"
            hashes[str(path)] = fingerprint_file(path).sha256
            ledgers.append([json.loads(line) for line in path.read_text().splitlines()])
        if any(len(ledger) != old.n_datasets for ledger in ledgers):
            raise ValueError("Incomplete paired dataset ledger")
        for i, (left, right) in enumerate(zip(*ledgers, strict=True)):
            name = f"dataset_{i:04d}"
            for row in (left, right):
                if row["case"] != case or row["repetition"] != i or row["directory"] != name:
                    raise ValueError("Paired row identity mismatch")
            if left["data_seed"] != right["data_seed"]:
                raise ValueError("Paired observed-data seeds differ")
            bases = [shard / name for shard in shards]
            raw = []
            for base in bases:
                files = {
                    p.name: fingerprint_file(p).sha256
                    for p in (base / "observed").iterdir()
                    if p.is_file()
                }
                raw.append(files)
            if raw[0] != raw[1]:
                raise ValueError("Paired raw observations are not byte-identical")
            outcomes, layouts = [], []
            for base, row in zip(bases, (left, right), strict=True):
                if row["outcome_file"] not in {"summary.json", "failure.json"}:
                    raise ValueError("Unknown paired outcome file")
                path = base / "calibration" / row["outcome_file"]
                if fingerprint_file(path).sha256 != row["outcome_sha256"]:
                    raise ValueError("Paired outcome checksum mismatch")
                hashes[str(path)] = row["outcome_sha256"]
                outcome = read_json(path)
                outcomes.append(outcome)
                layout = None
                if row["outcome_file"] == "summary.json":
                    path = base / "calibration/layout.json"
                    if fingerprint_file(path).sha256 != outcome["artifacts"]["layout.json"]:
                        raise ValueError("Paired fitted-layout checksum mismatch")
                    layout = read_json(path)
                layouts.append(layout)
            a, b = left["p_value"] is not None, right["p_value"] is not None
            ldiag = layouts[0]["diagnostics"] if layouts[0] is not None else {}
            rdiag = layouts[1]["diagnostics"] if layouts[1] is not None else {}
            row = {
                "case": case,
                "repetition": i,
                "data_seed": left["data_seed"],
                "raw_byte_identity_verified": True,
                "legacy_available": a,
                "current_available": b,
                "recovered": not a and b,
                "lost": a and not b,
                "legacy_p_value": left["p_value"],
                "current_p_value": right["p_value"],
                "legacy_statistic": left["statistic"],
                "current_statistic": right["statistic"],
                "legacy_k": left["selected_k"],
                "current_k": right["selected_k"],
                "conditional_rank": rdiag.get("conditional_rank"),
                "boundary_or_near_boundary": rdiag.get("boundary_or_near_boundary"),
                "legacy_nll": ldiag.get("negative_log_likelihood"),
                "current_nll": rdiag.get("negative_log_likelihood"),
            }
            row["point_statistic_difference"] = (
                right["statistic"] - left["statistic"] if a and b else None
            )
            row["nll_difference"] = (
                row["current_nll"] - row["legacy_nll"]
                if row["legacy_nll"] is not None and row["current_nll"] is not None
                else None
            )
            rows.append(row)
    totals = {
        key: sum(r[key] for r in rows)
        for key in ("legacy_available", "current_available", "recovered", "lost")
    }
    summary = {
        "schema_version": 1,
        "status": "complete",
        "paired_datasets": len(rows),
        **totals,
        "scope": (
            "Descriptive comparison on byte-identical observed cohorts "
            "and identical point settings; "
            "versioned null samplers differ, so bootstrap draws are not a common random bank"
        ),
        "raw_byte_identity_verified": True,
        "inputs_sha256": hashes,
    }
    destination.mkdir(exist_ok=False)
    with (destination / "datasets.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(summary, destination / "summary.json")
    return summary
