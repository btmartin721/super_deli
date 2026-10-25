"""Audit curated bytes and reviewed evidence; never dispatch a statistical fit."""

import csv
import hashlib
import io
from collections import Counter
from pathlib import Path, PurePosixPath

from super_deli.core.data import DatasetConfig, Feature, Modality, Observation, Specimen
from super_deli.core.empirical import EmpiricalProfile
from super_deli.core.empirical_evidence import EmpiricalEvidence
from super_deli.core.empirical_protocol import EmpiricalProtocol
from super_deli.core.provenance import configuration_hash
from super_deli.io.config import read_json, write_json
from super_deli.workflows.empirical_curation import collect_rows, fingerprint


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_curated(directory):
    """Checksums are integrity evidence only; semantic reconstruction follows."""
    root = Path(directory)
    manifest = read_json(root / "checksums.json")
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Curated checksum manifest required")
    files = {}
    for p in root.rglob("*"):
        if p.is_symlink():
            raise ValueError("Curated artifacts cannot contain symbolic links")
        if p.is_file():
            if p.stat().st_size > 8 * 1024**2:
                raise ValueError("Curated artifact exceeds 8 MiB")
            files[p.relative_to(root).as_posix()] = sha(p)
    if sum((root / n).stat().st_size for n in files) > 16 * 1024**2:
        raise ValueError("Curated directory exceeds 16 MiB")
    if set(files) != {*manifest, "checksums.json"}:
        raise ValueError("Unlisted, missing or incomplete curated artifacts")
    for name, digest in manifest.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or files.get(name) != digest:
            raise ValueError("Curated checksum mismatch or unsafe path")
    return files


def verify_evidence(evidence, directory):
    root = Path(directory)
    hashes = {}
    for s in evidence.sources:
        if s.size_bytes:
            p = root / (s.source_id + ".txt")
            if p.is_symlink() or p.stat().st_size != s.size_bytes or sha(p) != s.sha256:
                raise ValueError("Evidence snapshot differs from reviewed bytes")
            hashes[s.source_id] = s.sha256
    return hashes


def csv_bytes(header, rows):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def audit_cohort(curated, profile, protocol, records):
    root = Path(curated)
    if (
        read_json(root / "audit/profile.json") != profile.to_dict()
        or read_json(root / "audit/protocol.json") != protocol.to_dict()
    ):
        raise ValueError("Curated declarations differ from the bound protocol/profile")
    selected = [r for r in records if r["eligible"]]

    def check_table(name, columns, rows):
        if (root / name).read_bytes() != csv_bytes(columns, rows):
            raise ValueError("Curated source/identity/reference/feature mismatch: " + name)

    check_table(
        "audit/source_index.csv",
        ["specimen_id", "source_line", "included", *profile.identity_columns],
        [(r["unit"], r["source_line"], int(r["eligible"]), *r["key"]) for r in records],
    )
    if read_json(root / "audit/row_accounting.json") != [
        dict(
            specimen_id=r["unit"],
            source_line=r["source_line"],
            included=r["eligible"],
            missing_features=r["missing"],
        )
        for r in records
    ]:
        raise ValueError("Curated exclusion accounting differs from the source")
    check_table(
        "evaluation/references.csv",
        ["specimen_id", "reference_label"],
        [(r["unit"], r["row"][profile.reference_column]) for r in selected],
    )
    check_table(
        "evaluation/metadata.csv",
        ["specimen_id", *profile.metadata_columns],
        [(r["unit"], *(r["row"][c] for c in profile.metadata_columns)) for r in selected],
    )
    modalities = []
    for kind, name in (("morphological", "morphology"), ("ecological", "ecology")):
        features = [f for f in profile.features if f.kind == kind]
        observations = tuple(Observation(f"{name}_{r['unit']}", r["unit"]) for r in selected)
        check_table(
            "analysis/" + name + ".csv",
            ["observation_id", *(f.name for f in features)],
            [
                (o.observation_id, *(r["row"][f.source_column] for f in features))
                for o, r in zip(observations, selected, strict=True)
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
    expected = DatasetConfig(
        dataset_id=profile.dataset_id,
        modalities=tuple(modalities),
        specimens=tuple(Specimen(r["unit"]) for r in selected),
    )
    if read_json(root / "analysis/dataset.json") != expected.to_dict():
        raise ValueError("Curated manifest has altered units, links, roles or feature declarations")
    summary = read_json(root / "summary.json")
    if (
        summary.get("source") != profile.to_dict()["source"]
        or summary.get("readiness")
        != dict(
            ingestion=True,
            statistical_fitting=False,
            independence="unresolved",
            reference_independence="unresolved",
            grouped_predictive_screen="inapplicable_specimen_level",
            bootstrap="not_authorized",
            significance="not_authorized",
        )
        or summary.get("dataset_id") != profile.dataset_id
        or summary.get("ingestion_status") != "complete"
    ):
        raise ValueError("Curated source/readiness declarations differ from schema 1")
    counts = dict(
        source_rows=len(records),
        included_rows=len(selected),
        excluded_rows=len(records) - len(selected),
    )
    coverage = sum(
        r["row"][profile.reference_column] not in profile.missing_values for r in selected
    )
    if (
        summary.get("status") != "curated"
        or any(summary.get(k) != v for k, v in counts.items())
        or summary.get("reference_coverage") != coverage
    ):
        raise ValueError("Curated summary disagrees with source accounting")
    return dict(
        **counts,
        unique_source_keys=len({r["key"] for r in records}),
        distinct_identity_values={
            c: len({r["row"][c] for r in records}) for c in profile.identity_columns
        },
        reference_coverage=coverage,
        exact_numeric_tokens_checked=len(selected) * len(profile.features),
        reference_counts=dict(
            sorted(Counter(r["row"][profile.reference_column] for r in selected).items())
        ),
        metadata_counts={
            c: dict(sorted(Counter(r["row"][c] for r in selected).items()))
            for c in profile.metadata_columns
        },
        source_reference_counts=dict(
            sorted(Counter(r["row"][profile.reference_column] for r in records).items())
        ),
        source_metadata_counts={
            c: dict(sorted(Counter(r["row"][c] for r in records).items()))
            for c in profile.metadata_columns
        },
        identity_token_reuse={
            c: dict(
                distinct_reused_tokens=sum(
                    n > 1 for n in Counter(r["row"][c] for r in records).values()
                ),
                rows_with_reused_tokens=sum(
                    n for n in Counter(r["row"][c] for r in records).values() if n > 1
                ),
            )
            for c in profile.identity_columns
        },
        sampling_unit="capture_record",
        nest_or_persistent_bird_links_created=0,
    )


def readiness(evidence, facts):
    """Purpose-specific outcomes, not a general Boolean certificate."""
    claims = {c.claim_id: c for c in evidence.claims}

    def decision(status, reasons, claim_ids):
        return dict(status=status, reasons=reasons, evidence_claims=claim_ids)

    return dict(
        ingestion=decision("available", ["source_and_curated_semantics_verified"], []),
        published_reference_description=decision(
            "available" if facts["reference_coverage"] else "blocked",
            [
                "development_field_assignments_only"
                if facts["reference_coverage"]
                else "no_reference_coverage"
            ],
            ["reference_provenance"],
        ),
        independent_unit_point_fitting=decision(
            "blocked",
            ["fit_free_checkpoint", "independent_sampling_units_not_established"],
            [
                n
                for n in ("persistent_identity", "nest_crosswalk", "cross_season_identity")
                if claims[n].status == "not_established"
            ],
        ),
        grouped_predictive_screen=decision(
            "inapplicable",
            ["capture_level_inputs_without_verified_group_design"],
            ["nest_crosswalk"],
        ),
        resampling=decision(
            "blocked",
            ["no_authorized_independent_resampling_units", "fit_free_checkpoint"],
            ["nest_dependence", "nest_crosswalk", "cross_season_identity"],
        ),
        significance=decision(
            "blocked",
            ["no_approved_empirical_null_for_this_sampling_design", "fit_free_checkpoint"],
            ["nest_dependence", "persistent_identity"],
        ),
        independent_species_boundary_validation=decision(
            "blocked",
            [
                "independent_specimen_reference_not_established",
                "independent_sampling_units_not_established",
            ],
            ["specimen_reference_independence", "taxonomic_context"],
        ),
    )


def assess_empirical(
    source_path, *, profile, protocol, evidence, curated_dir, evidence_dir, output_dir
):
    profile = EmpiricalProfile.from_dict(profile.to_dict())
    protocol = EmpiricalProtocol.from_dict(protocol.to_dict())
    evidence = EmpiricalEvidence.from_dict(evidence.to_dict())
    evidence.check_bindings(profile, protocol)
    records = collect_rows(source_path, profile)
    initial = verify_curated(curated_dir)
    snapshots = verify_evidence(evidence, evidence_dir)
    facts = audit_cohort(curated_dir, profile, protocol, records)
    out = Path(output_dir).absolute()
    # Never write assessment artifacts inside the preserved inputs.
    protected = [Path(p).resolve() for p in (curated_dir, evidence_dir)]
    if any(
        out.resolve().is_relative_to(p) or p.is_relative_to(out.resolve()) for p in protected
    ) or Path(source_path).resolve().is_relative_to(out.resolve()):
        raise ValueError("Assessment output must be separate from all inputs")
    out.mkdir(parents=True, exist_ok=False)
    try:
        write_json(evidence.to_dict(), out / "reviewed-evidence.json")
        write_json(
            dict(curated_files=initial, evidence_snapshots=snapshots), out / "input-integrity.json"
        )
        write_json(facts, out / "source-facts.json")
        result = dict(
            schema_version=1,
            status="assessed",
            phase="readiness_audit_only",
            dataset_id=profile.dataset_id,
            source_sha256=profile.source.sha256,
            profile_sha256=configuration_hash(profile.to_dict()),
            protocol_sha256=configuration_hash(protocol.to_dict()),
            evidence_sha256=configuration_hash(evidence.to_dict()),
            source_rows=facts["source_rows"],
            included_rows=facts["included_rows"],
            reference_coverage=facts["reference_coverage"],
            readiness=readiness(evidence, facts),
            empirical_fits=0,
            stability_draws=0,
            significance_draws=0,
            review_boundary=(
                "Hashes bind snapshots and declarations; reviewed prose is not "
                "machine-certified biological truth"
            ),
            requirements={
                c.claim_id: c.required_evidence
                for c in evidence.claims
                if c.status == "not_established"
            },
        )
        fingerprint(source_path, profile.source)
        if (
            verify_curated(curated_dir) != initial
            or verify_evidence(evidence, evidence_dir) != snapshots
        ):
            raise ValueError("Inputs changed during assessment")
        pending = out / "summary.pending.json"
        write_json(result, pending)
        hashes = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p != pending}
        hashes["summary.json"] = sha(pending)
        write_json(hashes, out / "checksums.json")
        pending.rename(out / "summary.json")
        return result
    except Exception as exc:
        write_json(dict(status="failed", error=str(exc)), out / "failure.json")
        raise
