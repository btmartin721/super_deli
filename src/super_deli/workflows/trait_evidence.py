"""Source-bound trait evidence assessment with no statistical dispatch."""

import gzip
import hashlib
from collections import Counter
from pathlib import Path

from super_deli.core.provenance import configuration_hash
from super_deli.core.trait_audit import TraitAuditPlan
from super_deli.core.trait_evidence import TraitEvidencePlan
from super_deli.io.config import write_json
from super_deli.io.trait_evidence import (
    accession_base,
    assess_records,
    nexus_context,
    reference_tables,
    source_accounting,
    verify_source_inventories,
)
from super_deli.workflows.correspondence import verify_evidence
from super_deli.workflows.empirical_assessment import sha, verify_curated
from super_deli.workflows.trait_audit import audit_trait_source, verify_trait_sources, write_gz_rows


def _ledger(path):
    import json

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def verify_prior(plan, prior, source_dir, audit_dir, reconstruction):
    if configuration_hash(prior.to_dict()) != plan.prior_specification_sha256:
        raise ValueError("Prior trait specification differs from the bound audit")
    root = Path(audit_dir)
    if root.is_symlink() or sha(root / "checksums.json") != plan.prior_checksums_sha256:
        raise ValueError("Prior trait artifact inventory differs")
    before = verify_curated(root)
    paths, source_hashes = verify_trait_sources(prior, source_dir)
    audit_trait_source(prior, source_dir=source_dir, output_dir=reconstruction)
    rebuilt = verify_curated(reconstruction)
    if before != rebuilt:
        differing = sorted(
            k for k in before.keys() | rebuilt.keys() if before.get(k) != rebuilt.get(k)
        )
        raise ValueError(
            "Prior semantics differ from complete source reconstruction: " + repr(differing)
        )
    return dict(
        source_hashes=source_hashes, artifact_hashes=before, full_semantic_reconstruction=True
    ), paths


def reviewed_claims(plan, paths):
    result = []
    for claim in plan.claims:
        document = paths[claim.source_id].read_text(encoding="utf-8-sig")
        if claim.anchor not in document:
            raise ValueError("Reviewed source anchor absent: " + claim.claim_id)
        result.append(
            dict(
                claim_id=claim.claim_id,
                status=claim.status,
                source_id=claim.source_id,
                locator=claim.locator,
                anchor=claim.anchor,
                interpretation=claim.interpretation,
                limitation=claim.limitation,
                required_evidence=claim.required_evidence,
                evidence_type="reviewed_claim_not_computed_truth",
            )
        )
    return result


def readiness(records):
    linked = sum(r["molecular_partition_link"] for r in records)
    return dict(
        ingestion=dict(status="available", reason="unchanged_prior_ingestion_verified"),
        descriptive_reference_labels=dict(status="available", reason="mixed_source_basis_retained"),
        molecular_reference_concordance=dict(
            status="documented_links_available_for_future_design" if linked else "unresolved",
            documented_rows=linked,
            reason="methodological_partitions_are_not_species_truth",
            fitting_authorized=False,
        ),
        independent_unit_fitting=dict(
            status="blocked", reason="independent_units_not_established_and_fit_free_scope"
        ),
        grouped_predictive_screen=dict(
            status="blocked", reason="no_verified_independent_group_design_or_continuous_ecology"
        ),
        resampling=dict(status="blocked", reason="no_approved_independent_resampling_units"),
        significance=dict(
            status="blocked", reason="no_validated_empirical_null_for_this_sampling_design"
        ),
        species_boundary_validation=dict(
            status="blocked", reason="species_truth_and_sampling_independence_not_established"
        ),
        catalogue_crosswalk=dict(
            status="unavailable_publicly", reason="catalogue_access_statement"
        ),
        historical_motu_replacement=dict(
            status="not_authorized", reason="retain_distinct_source_versions_and_partitions"
        ),
    )


def assess_trait_evidence(
    plan, *, prior_plan, source_dir, prior_audit_dir, evidence_dir, output_dir
):
    plan = TraitEvidencePlan.from_dict(plan.to_dict())
    prior = TraitAuditPlan.from_dict(prior_plan.to_dict())
    out = Path(output_dir).absolute()
    for directory in (source_dir, prior_audit_dir, evidence_dir):
        root = Path(directory).resolve()
        if out.resolve().is_relative_to(root) or root.is_relative_to(out.resolve()):
            raise ValueError("Assessment outputs must be separate from all input roots")
    out.mkdir(parents=True, exist_ok=False)
    try:
        write_json(plan.to_dict(), out / "specification.json")
        paths, before = verify_evidence(plan, evidence_dir)
        integrity, _ = verify_prior(
            plan, prior, source_dir, prior_audit_dir, out / "prior-reconstruction"
        )
        verify_source_inventories(paths)
        claims = reviewed_claims(plan, paths)
        books, tables = reference_tables(paths)
        rows = _ledger(Path(prior_audit_dir) / "source-records.jsonl.gz")
        old_links = _ledger(Path(prior_audit_dir) / "reference-links.jsonl.gz")
        locations = _ledger(Path(prior_audit_dir) / "locality-links.jsonl.gz")
        context = nexus_context(paths.get("tree"), paths.get("regional"))
        records = assess_records(rows, old_links, locations, tables, context)
        facts = source_accounting(tables)
        write_gz_rows(out / "specimen-evidence.jsonl.gz", records)
        write_gz_rows(out / "reference-records.jsonl.gz", (r for rs in tables.values() for r in rs))
        write_gz_rows(
            out / "reference-cells.jsonl.gz",
            (
                dict(source_id=source_id, sheet=sheet, **row)
                for source_id, book in books.items()
                for sheet, sheet_rows in book["sheets"].items()
                for row in sheet_rows
            ),
        )
        write_json(
            {key: {k: v for k, v in book.items() if k != "sheets"} for key, book in books.items()},
            out / "reference-workbooks.json",
        )
        write_json(context, out / "phylogeny-context.json")
        write_json(claims, out / "reviewed-claims.json")
        write_json(facts, out / "source-accounting.json")
        write_json(dict(prior=integrity, new_source_hashes=before), out / "input-integrity.json")
        counts = Counter(r["correspondence_status"] for r in records)
        summary = dict(
            schema_version=1,
            phase=plan.phase,
            assessment_id=plan.assessment_id,
            specification_sha256=configuration_hash(plan.to_dict()),
            status="assessed",
            source_rows=len(records),
            valid_stated_accessions=sum(
                accession_base(r["stated_accession"]) is not None for r in records
            ),
            correspondence_counts=dict(sorted(counts.items())),
            documented_specimen_links=sum(r["documented_specimen_link"] for r in records),
            molecular_partition_links=sum(r["molecular_partition_link"] for r in records),
            explicit_table_corroborations=counts["corroborated_explicit_table_pair"],
            source_basis_counts=dict(
                sorted(Counter(x for r in records for x in r["source_basis"]).items())
            ),
            composite_context_rows=sum(bool(r["composite_multilocus_records"]) for r in records),
            partition_relations_on_linked_rows=dict(
                sorted(
                    Counter(
                        r["partition_relation"] for r in records if r["molecular_partition_link"]
                    ).items()
                )
            ),
            source_accounting=facts,
            published_vs_retained_counts={
                "delimitation_source_rows": dict(
                    paper=3310, retained=facts["reference01"]["records"]
                ),
                "phylogeny_source_rows": dict(paper=561, retained=facts["reference02"]["records"]),
                "spatial_source_rows": dict(paper=376, retained=facts["reference03"]["records"]),
            },
            count_comparison_scope=(
                "Reported study counts versus retained table rows; "
                "no biological-unit equivalence assumed"
            ),
            source_workbook_rows=sum(
                x["serialized_rows"] for b in books.values() for x in b["worksheets"]
            ),
            empty_formatting_rows=sum(
                x["empty_rows"] for b in books.values() for x in b["worksheets"]
            ),
            readiness=readiness(records),
            cohort_created=False,
            empirical_fits=0,
            bootstrap_draws=0,
            pvalues_generated=0,
            source_code_executed=False,
            contact_performed=False,
            private_catalogue_access=False,
            historical_partitions_replaced=False,
            sampling_independence_asserted=False,
            reference_species_truth_asserted=False,
        )
        _, after = verify_evidence(plan, evidence_dir)
        if before != after or verify_curated(prior_audit_dir) != integrity["artifact_hashes"]:
            raise ValueError("Evidence or prior artifacts changed during assessment")
        verify_trait_sources(prior, source_dir)
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
