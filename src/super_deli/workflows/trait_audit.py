"""Fit-free source auditing; no estimator or scientific-library imports."""

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from super_deli.core.data import (
    DatasetConfig,
    Feature,
    Modality,
    Observation,
    Specimen,
    TabularReaderOptions,
)
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.trait_audit import COLUMNS, TRAITS, TraitAuditPlan
from super_deli.core.validation import validate_dataset
from super_deli.io.audit_workbook import inspect_workbook
from super_deli.io.config import read_json, write_json
from super_deli.io.tabular import _read_table as read_table
from super_deli.io.trait_source import (
    ACCESSION,
    MISSING,
    inspect_rows,
    locality_accounting,
    molecular_records,
    reference_accounting,
    source_rows,
    trait_profiles,
)
from super_deli.workflows.empirical_curation import write_rows


def verify_trait_sources(plan, directory):
    root = Path(directory)
    if root.is_symlink():
        raise ValueError("Source root cannot be a symbolic link")
    paths, hashes = {}, {}
    for s in plan.sources:
        p = root / s.filename
        if p.is_symlink():
            raise ValueError("Source snapshots cannot follow symbolic links")
        if s.size_bytes:
            if not p.is_file() or p.stat().st_size != s.size_bytes:
                raise ValueError("Source size differs: " + s.role)
            fp = fingerprint_file(p)
            if fp.sha256 != s.sha256:
                raise ValueError("Source hash differs: " + s.role)
            hashes[s.role] = fp.sha256
        elif p.exists():
            raise ValueError("Unexpected bytes for an absent snapshot")
        if s.access == "usable":
            paths[s.role] = p
    return paths, hashes


def write_gz_rows(path, rows):
    with (
        path.open("xb") as raw,
        gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as zipped,
    ):
        for row in rows:
            zipped.write(
                (
                    json.dumps(
                        row,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
            )


def archive_integrity(paths):
    archive = read_json(paths["archive"])
    files = archive.get("files") if isinstance(archive, dict) else None
    if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], dict):
        raise ValueError("Exactly one versioned archive workbook required")
    license_info = archive.get("license")
    if not isinstance(license_info, dict) or license_info.get("name") != "CC0":
        raise ValueError("Explicit CC0 source archive required")
    raw = paths["workbook"].read_bytes()
    if (
        files[0].get("size") != len(raw)
        or files[0].get("computed_md5") != hashlib.md5(raw, usedforsecurity=False).hexdigest()
    ):
        raise ValueError("Archive inventory differs from workbook bytes")
    collection = read_json(paths["collection"])
    if (
        not isinstance(collection, list)
        or sum(
            isinstance(x, dict)
            and x.get("id") == archive.get("id")
            and x.get("doi") == archive.get("doi")
            for x in collection
        )
        != 1
    ):
        raise ValueError("Versioned workbook is not uniquely bound to its source collection")
    return {
        k: archive.get(k)
        for k in ("id", "doi", "version", "published_date", "modified_date", "license")
    }


def export_representation(rows, directory, audit_id):
    eligible = sorted((r for r in rows if r["representation_eligible"]), key=lambda r: r["unit_id"])
    if not eligible:
        return dict(
            status="blocked_no_unique_numeric_representation",
            observations=0,
            finite_tokens=0,
            missing_tokens=0,
        )
    directory.mkdir()
    write_rows(
        directory / "morphology.csv",
        ["observation_id", *TRAITS],
        [("m_" + r["unit_id"], *(r["values"][name] for name in TRAITS)) for r in eligible],
    )
    write_rows(
        directory / "source-index.csv",
        ["observation_id", "unit_id", "source_row"],
        [("m_" + r["unit_id"], r["unit_id"], r["source_row"]) for r in eligible],
    )
    m = Modality(
        name="morphology",
        kind="morphological",
        level="specimen",
        format="csv",
        path="morphology.csv",
        observations=tuple(Observation("m_" + r["unit_id"], r["unit_id"]) for r in eligible),
        features=tuple(Feature(name, "continuous", "mm") for name in TRAITS),
        tabular_options=TabularReaderOptions(missing_values=("", "NA", "unknown")),
    )
    parsed = read_table(directory / "morphology.csv", m)
    assert parsed.observation_ids == tuple(o.observation_id for o in m.observations)
    config = DatasetConfig(
        dataset_id=audit_id,
        modalities=(m,),
        specimens=tuple(Specimen(r["unit_id"]) for r in eligible),
    )
    report = validate_dataset(config, base_dir=directory)
    if not report.valid:
        raise ValueError("Existing-format representation failed structural validation")
    write_json(config.to_dict(), directory / "dataset.json")
    write_json(report.to_dict(), directory / "validation.json")
    return dict(
        status="available_ingestion_only",
        observations=len(eligible),
        finite_tokens=sum(r["numeric_values"][n] is not None for r in eligible for n in TRAITS),
        missing_tokens=sum(r["numeric_values"][n] is None for r in eligible for n in TRAITS),
        numeric_values_recalculated=False,
        statistical_cohort_authorized=False,
    )


def audit_trait_source(plan, *, source_dir, output_dir):
    plan = TraitAuditPlan.from_dict(plan.to_dict())
    paths, before = verify_trait_sources(plan, source_dir)
    source_root = Path(source_dir).resolve()
    out = Path(output_dir).absolute()
    if out.resolve().is_relative_to(source_root) or source_root.is_relative_to(out.resolve()):
        raise ValueError("Audit outputs must be separate from source snapshots")
    out.mkdir(parents=True, exist_ok=False)
    try:
        write_json(plan.to_dict(), out / "specification.json")
        archive = archive_integrity(paths)
        screen = read_json(paths["screen"])
        if (
            not isinstance(screen, dict)
            or screen.get("contact_performed") is not False
            or screen.get("selection_basis") != "traceability_and_compatibility"
            or not isinstance(screen.get("candidates"), list)
            or not screen["candidates"]
        ):
            raise ValueError("Explicit source-based screen without contact required")
        book = inspect_workbook(paths["workbook"])
        rows, dictionary, header = source_rows(book)
        inspect_rows(rows)
        refs = molecular_records(paths["references"]) if "references" in paths else []
        links = reference_accounting(rows, refs, "references" in paths)
        locations, contexts, location_summary = locality_accounting(rows, book)
        write_gz_rows(
            out / "cells.jsonl.gz",
            (dict(sheet=sheet, **r) for sheet, data in book["sheets"].items() for r in data),
        )
        write_gz_rows(
            out / "source-records.jsonl.gz",
            ({k: v for k, v in r.items() if k != "cells"} for r in rows),
        )
        write_gz_rows(out / "reference-links.jsonl.gz", links)
        write_gz_rows(out / "molecular-records.jsonl.gz", refs)
        write_gz_rows(out / "locality-links.jsonl.gz", locations)
        write_gz_rows(out / "context-descriptors.jsonl.gz", contexts)
        write_json({k: v for k, v in book.items() if k != "sheets"}, out / "workbook-metadata.json")
        write_json(dictionary, out / "data-dictionary.json")
        write_json(trait_profiles(rows), out / "trait-profiles.json")
        write_json(
            {
                name: dict(
                    nonmissing=sum(r["values"][name] not in MISSING for r in rows),
                    distinct_nonmissing=len(
                        {r["values"][name] for r in rows if r["values"][name] not in MISSING}
                    ),
                    missing_tokens=dict(
                        sorted(
                            Counter(
                                r["values"][name] for r in rows if r["values"][name] in MISSING
                            ).items()
                        )
                    ),
                    stored_types=dict(
                        sorted(
                            Counter(
                                r["cells"].get(name, {}).get("type", "absent") for r in rows
                            ).items()
                        )
                    ),
                )
                for name in COLUMNS
            },
            out / "column-profiles.json",
        )
        representation = export_representation(rows, out / "analysis", plan.audit_id)
        write_json(representation, out / "representation.json")
        write_json(screen, out / "source-screen.json")
        computed = dict(
            rows=len(rows),
            motu_labels=len(
                {r["values"]["MOTU"] for r in rows if r["values"]["MOTU"] not in MISSING}
            ),
            locality_names=len(
                {
                    r["values"]["locality_name"]
                    for r in rows
                    if r["values"]["locality_name"] not in MISSING
                }
            ),
        )
        formulas = [
            dict(source_row=r["source_row"], field=n, cell=c)
            for r in rows
            for n, c in r["cells"].items()
            if c["formula"] is not None
        ]
        write_json(formulas, out / "formula-provenance.json")
        requested = {
            r["values"]["COI_sequence"].split(".")[0]
            for r in rows
            if ACCESSION.fullmatch(r["values"]["COI_sequence"])
        }
        returned = {r["accession"] for r in refs}
        summary = dict(
            schema_version=1,
            status="audited",
            phase=plan.phase,
            audit_id=plan.audit_id,
            specification_sha256=configuration_hash(plan.to_dict()),
            archive=archive,
            source_rows=len(rows),
            source_columns=len(COLUMNS),
            continuous_traits=len(TRAITS),
            identity_counts=dict(sorted(Counter(r["identity_status"] for r in rows).items())),
            legacy_identifiers_are_unique=False,
            valid_accession_rows=sum(
                bool(ACCESSION.fullmatch(r["values"]["COI_sequence"])) for r in rows
            ),
            requested_accessions=len(requested),
            returned_reference_records=len(refs),
            missing_requested_accessions=len(requested - returned),
            unexpected_reference_records=len(returned - requested),
            reference_link_counts=dict(sorted(Counter(link["status"] for link in links).items())),
            unflagged_literal_links=sum(link["unflagged_literal_link"] for link in links),
            source_reference_caution_rows=sum(bool(r["reference_cautions"]) for r in rows),
            accession_rows_with_source_cautions=sum(
                bool(r["reference_cautions"])
                and bool(ACCESSION.fullmatch(r["values"]["COI_sequence"]))
                for r in rows
            ),
            numeric_issue_rows=sum(bool(r["numeric_issues"]) for r in rows),
            nonpositive_measurement_rows=sum(bool(r["nonpositive_traits"]) for r in rows),
            formula_cells=len(formulas),
            formula_evaluation=False,
            representation=representation,
            locality_accounting=location_summary,
            published_counts=dict(plan.published_counts),
            computed_counts=computed,
            count_disagreements={
                k: dict(published=v, computed=computed[k])
                for k, v in plan.published_counts
                if v != computed[k]
            },
            readiness=dict(
                continuous_ingestion=representation["status"],
                specimen_traceability="available_for_unique_source_vouchers_only"
                if any(r["identity_status"] == "unique" for r in rows)
                else "unavailable_no_unique_source_vouchers",
                direct_molecular_correspondence="available_for_unflagged_literal_links_only"
                if any(link["unflagged_literal_link"] for link in links)
                else "unavailable_no_unflagged_literal_links",
                reference_label_independence="not_established_by_DNA_availability",
                biological_independence="requires_separate_sampling_design",
                ecological_model="not_authorized_source_categories_are_metadata",
                genomic_model="not_authorized_COI_is_reference_only",
                empirical_fitting="not_authorized",
                bootstrap="not_authorized",
                significance="not_authorized",
            ),
            empirical_fits=0,
            pvalues_generated=0,
            bootstrap_draws=0,
            contact_performed=False,
            email_sent=False,
        )
        _, after = verify_trait_sources(plan, source_dir)
        if after != before:
            raise ValueError("Source snapshots changed during audit")
        pending = out / "summary.pending.json"
        write_json(summary, pending)
        hashes = {
            p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob("*"))
            if p.is_file() and p != pending
        }
        hashes["summary.json"] = hashlib.sha256(pending.read_bytes()).hexdigest()
        write_json(hashes, out / "checksums.json")
        pending.rename(out / "summary.json")
        return summary
    except Exception as exc:
        write_json(dict(status="failed", error=str(exc)), out / "failure.json")
        raise
