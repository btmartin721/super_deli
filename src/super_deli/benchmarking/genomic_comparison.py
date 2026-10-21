"""Byte-verified paired comparison of pooled-HWE and block-profile genomic tests."""

import csv
from pathlib import Path

from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json
from super_deli.models.genomic_config import GenomicStudyConfig
from super_deli.models.linked_config import PAIRED_CASES, LinkedStudyConfig


def compare_genomic_studies(legacy, current, *, output_dir):
    legacy, current, destination = (
        Path(legacy).resolve(),
        Path(current).resolve(),
        Path(output_dir).resolve(),
    )
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    if any(destination.is_relative_to(p) for p in (legacy, current)):
        raise ValueError("Comparison must be outside immutable study shards")
    rows = []
    for case in PAIRED_CASES:
        old = GenomicStudyConfig.from_dict(read_json(legacy / case / "study.json"))
        new = LinkedStudyConfig.from_dict(read_json(current / case / "study.json"))
        if (old.n_units, old.n_datasets, old.random_seed) != (
            new.n_units,
            new.n_datasets,
            new.random_seed,
        ):
            raise ValueError("Paired study designs/seeds disagree")
        a, b = old.calibration.model.to_dict(), new.calibration.model.to_dict()
        for key in ("assume_unlinked", "missing_policy"):
            a.pop(key)
            b.pop(key)
        if a != b or old.calibration.alpha != new.calibration.alpha:
            raise ValueError("Paired point settings/alpha disagree")
        left = read_json(legacy / case / "summary.json")["datasets"]
        right = read_json(current / case / "summary.json")["datasets"]
        if len(left) != len(right) or len(left) != old.n_datasets:
            raise ValueError("Paired dataset counts disagree")
        for i, (x, y) in enumerate(zip(left, right, strict=True)):
            for key in ("case", "repetition", "data_seed", "null_stream_seed", "values_sha256"):
                if x[key] != y[key]:
                    raise ValueError(f"Paired {key} mismatch")
            for name in ("genotypes.phy", "dataset.json"):
                if (
                    fingerprint_file(legacy / case / x["directory"] / "observed" / name).sha256
                    != fingerprint_file(current / case / y["directory"] / "observed" / name).sha256
                ):
                    raise ValueError("Paired raw bytes disagree")
            diff = None
            if x["statistic"] is not None and y["statistic"] is not None:
                diff = y["statistic"] - x["statistic"]
                if abs(diff) > 1e-8 or x["selected_k"] != y["selected_k"]:
                    raise ValueError("Paired observed pipeline statistics/K disagree")
            rows.append(
                {
                    "case": case,
                    "repetition": i,
                    "data_seed": x["data_seed"],
                    "legacy_available": x["p_value"] is not None,
                    "current_available": y["p_value"] is not None,
                    "legacy_p_value": x["p_value"],
                    "current_p_value": y["p_value"],
                    "legacy_reject": x["reject"],
                    "current_reject": y["reject"],
                    "legacy_k": x["selected_k"],
                    "current_k": y["selected_k"],
                    "point_statistic_difference": diff,
                    "raw_byte_identity_verified": True,
                }
            )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "paired_datasets": len(rows),
        "raw_byte_identity_verified": True,
        "legacy_available": sum(r["legacy_available"] for r in rows),
        "current_available": sum(r["current_available"] for r in rows),
        "lost": sum(r["legacy_available"] and not r["current_available"] for r in rows),
        "recovered": sum(not r["legacy_available"] and r["current_available"] for r in rows),
        "scope": (
            "Complete shared observations and point settings; pooled-HWE and "
            "empirical-block nulls impose different hypotheses and use different null"
            " seed namespaces. P-value differences are not a common-random-bank "
            "comparison."
        ),
    }
    destination.mkdir()
    with (destination / "datasets.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(summary, destination / "summary.json")
    return summary
