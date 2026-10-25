"""Curate matched source rows without importing estimators or asserting independence."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from super_deli.core.data import DatasetConfig, Feature, Modality, Observation, Specimen
from super_deli.core.empirical import EmpiricalProfile
from super_deli.core.empirical_protocol import EmpiricalProtocol
from super_deli.core.validation import validate_dataset
from super_deli.io.config import write_json
from super_deli.io.delimited import delimited_rows, identifier
from super_deli.io.numeric import finite_float
from super_deli.io.tabular import ingest_tabular


def fingerprint(path, source):
    path = Path(path)
    with path.open("rb") as handle:
        raw = handle.read(source.size_bytes + 1)
    if len(raw) != source.size_bytes or hashlib.sha256(raw).hexdigest() != source.sha256:
        raise ValueError("Input bytes differ from the frozen source size/SHA-256")
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if blob != source.git_blob_sha1:
        raise ValueError("Input Git blob identity differs")


def write_rows(path, header, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def collect_rows(path, profile):
    fingerprint(path, profile.source)
    records, keys, ids = [], set(), set()
    for line, row in delimited_rows(
        Path(path), delimiter=",", required=set(profile.source_columns), context="empirical source"
    ):
        key = tuple(
            identifier(row[c], f"source line {line} identity") for c in profile.identity_columns
        )
        if any(v in profile.missing_values for v in key) or key in keys:
            raise ValueError("Missing or duplicate source identity; no automatic deduplication")
        keys.add(key)
        encoded = json.dumps(
            [profile.dataset_id, *key], ensure_ascii=True, separators=(",", ":")
        ).encode()
        unit = "unit_" + hashlib.sha256(encoded).hexdigest()[:24]
        if unit in ids:
            raise ValueError("Anonymous identifier collision")
        ids.add(unit)
        missing = []
        for feature in profile.features:
            token = row[feature.source_column]
            if token in profile.missing_values:
                missing.append(feature.name)
                continue
            if token != token.strip():
                raise ValueError("Padded numeric tokens cannot be normalized implicitly")
            value = finite_float(token, f"source line {line}: {feature.source_column}")
            if feature.positive and value <= 0:
                raise ValueError("Physically positive measurements must exceed zero")
        records.append(
            dict(
                unit=unit, key=key, source_line=line, row=row, missing=missing, eligible=not missing
            )
        )
    if not records or not any(r["eligible"] for r in records):
        raise ValueError("At least one complete matched source record is required")
    fingerprint(path, profile.source)
    return sorted(records, key=lambda r: r["unit"])


def curate_empirical(source_path, *, profile, protocol, output_dir):
    profile = EmpiricalProfile.from_dict(profile.to_dict())
    protocol = EmpiricalProtocol.from_dict(protocol.to_dict())
    protocol.check_profile(profile)
    records = collect_rows(source_path, profile)
    selected = [r for r in records if r["eligible"]]
    out = Path(output_dir).absolute()
    out.mkdir(parents=True, exist_ok=False)
    try:
        for name in ("analysis", "evaluation", "audit"):
            (out / name).mkdir()
        write_json(profile.to_dict(), out / "audit/profile.json")
        write_json(protocol.to_dict(), out / "audit/protocol.json")
        modalities = []
        for kind, name in (("morphological", "morphology"), ("ecological", "ecology")):
            features = tuple(f for f in profile.features if f.kind == kind)
            observations = tuple(Observation(f"{name}_{r['unit']}", r["unit"]) for r in selected)
            write_rows(
                out / "analysis" / (name + ".csv"),
                ["observation_id", *(f.name for f in features)],
                [
                    (o.observation_id, *(r["row"][f.source_column] for f in features))
                    for r, o in zip(selected, observations, strict=True)
                ],
            )
            modalities.append(
                Modality(
                    name=name,
                    kind=kind,
                    level="specimen",
                    format="csv",
                    path=name + ".csv",
                    observations=observations,
                    features=tuple(Feature(f.name, "continuous", f.unit) for f in features),
                )
            )
        dataset = DatasetConfig(
            dataset_id=profile.dataset_id,
            modalities=tuple(modalities),
            specimens=tuple(Specimen(r["unit"]) for r in selected),
        )
        report = validate_dataset(dataset, base_dir=out / "analysis")
        if not report.valid:
            raise ValueError("Curated dataset declaration failed existing structural validation")
        write_json(dataset.to_dict(), out / "analysis/dataset.json")
        ingestion = ingest_tabular(
            dataset, base_dir=out / "analysis", output_dir=out / "audit/ingestion"
        )
        write_rows(
            out / "evaluation/references.csv",
            ["specimen_id", "reference_label"],
            [(r["unit"], r["row"][profile.reference_column]) for r in selected],
        )
        write_rows(
            out / "evaluation/metadata.csv",
            ["specimen_id", *profile.metadata_columns],
            [(r["unit"], *(r["row"][c] for c in profile.metadata_columns)) for r in selected],
        )
        write_rows(
            out / "audit/source_index.csv",
            ["specimen_id", "source_line", "included", *profile.identity_columns],
            [(r["unit"], r["source_line"], int(r["eligible"]), *r["key"]) for r in records],
        )
        write_json(
            [
                dict(
                    specimen_id=r["unit"],
                    source_line=r["source_line"],
                    included=r["eligible"],
                    missing_features=r["missing"],
                )
                for r in records
            ],
            out / "audit/row_accounting.json",
        )
        fingerprint(source_path, profile.source)
        missing_counts = Counter(f for r in records for f in r["missing"])
        result = dict(
            schema_version=1,
            status="curated",
            dataset_id=profile.dataset_id,
            source_rows=len(records),
            included_rows=len(selected),
            excluded_rows=len(records) - len(selected),
            missing_features={f.name: missing_counts[f.name] for f in profile.features},
            reference_coverage=sum(
                r["row"][profile.reference_column] not in profile.missing_values for r in selected
            ),
            source=profile.to_dict()["source"],
            matching="Explicit same-row identity columns; no cross-source inference",
            identity_grain=profile.identity_grain,
            validation=report.to_dict(),
            ingestion_status=ingestion.summary["status"],
            readiness=dict(
                ingestion=True,
                statistical_fitting=False,
                independence="unresolved",
                reference_independence="unresolved",
                grouped_predictive_screen="inapplicable_specimen_level",
                bootstrap="not_authorized",
                significance="not_authorized",
            ),
            transformations="none; numeric tokens preserved; complete-feature cohort only",
        )
        pending = out / "summary.pending.json"
        write_json(result, pending)
        hashes = {
            p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob("*"))
            if p.is_file() and p != pending
        }
        hashes["summary.json"] = hashlib.sha256(pending.read_bytes()).hexdigest()
        write_json(hashes, out / "checksums.json")
        pending.rename(out / "summary.json")
        return result
    except Exception as exc:
        write_json(dict(status="failed", error=str(exc)), out / "failure.json")
        raise
