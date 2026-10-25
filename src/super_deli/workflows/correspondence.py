"""Resolve reviewed documentary correspondence without analytical dispatch.

Source-role review is a curator responsibility. Integrity and exact source-row
checks cannot authenticate a document's issuer or establish biological truth.
"""

from collections import Counter, defaultdict
from pathlib import Path

from super_deli.core.correspondence import CorrespondencePlan
from super_deli.core.provenance import configuration_hash
from super_deli.core.replacement_audit import ReplacementAuditPlan
from super_deli.io.config import read_json, write_json
from super_deli.io.correspondence import (
    HAUL_COLUMNS,
    HAUL_KEY,
    VOUCHER_COLUMNS,
    crosswalk_rows,
    field_dictionary,
    haul_catalogue,
)
from super_deli.io.specimen_audit import genbank_references, source_tables
from super_deli.workflows.empirical_assessment import sha, verify_curated
from super_deli.workflows.empirical_curation import write_rows
from super_deli.workflows.replacement_audit import reference_links, verify_sources


def verify_evidence(plan, directory):
    root = Path(directory)
    paths, hashes = {}, {}
    if root.is_symlink():
        raise ValueError("Evidence root cannot be a symbolic link")
    for source in plan.sources:
        p = root / source.filename
        if (
            p.is_symlink()
            or not p.is_file()
            or p.stat().st_size != source.size_bytes
            or sha(p) != source.sha256
        ):
            raise ValueError("Reviewed evidence size/hash mismatch: " + source.source_id)
        paths[source.source_id], hashes[source.source_id] = p, source.sha256
    return paths, hashes


def verify_prior(plan, prior, source_dir, audit_dir):
    if configuration_hash(prior.to_dict()) != plan.prior_specification_sha256:
        raise ValueError("Prior specification differs from the bound audit")
    root = Path(audit_dir)
    if root.is_symlink() or sha(root / "checksums.json") != plan.prior_checksums_sha256:
        raise ValueError("Prior artifact inventory differs from the bound audit")
    artifacts = verify_curated(root)
    if read_json(root / "specification.json") != prior.to_dict():
        raise ValueError("Prior source declarations differ")
    paths, hashes = verify_sources(prior, source_dir)
    metadata, _ = source_tables(paths["metadata"], paths["colour"])
    refs = genbank_references(paths["references"])
    links = reference_links(metadata, refs, prior)
    if (
        read_json(root / "reference-links.json") != links
        or read_json(root / "molecular-records.json") != refs
    ):
        raise ValueError("Prior reference ledger differs from source reconstruction")
    summary = read_json(root / "summary.json")
    if (
        summary.get("status") != "audited"
        or summary.get("empirical_fits") != 0
        or summary.get("specification_sha256") != plan.prior_specification_sha256
        or summary.get("metadata_records") != len(metadata)
        or summary.get("reference_records") != len(refs)
    ):
        raise ValueError("Prior summary is incomplete or inconsistent")
    return metadata, refs, links, dict(source_hashes=hashes, artifact_hashes=artifacts)


def documentary_claims(plan, paths):
    claims = {"voucher_crosswalk": [], "haul_crosswalk": []}
    for source in plan.sources:
        if source.role in claims:
            for entry in crosswalk_rows(paths[source.source_id], source.role):
                claims[source.role].append(
                    dict(
                        **entry,
                        source_id=source.source_id,
                        source_sha256=source.sha256,
                        provenance_group=source.provenance_group,
                        locator="CSV line " + str(entry["line"]),
                        review_locator=source.review_locator,
                    )
                )
    return claims


def resolve_links(metadata, refs, candidates, claims, catalogue):
    """Retain conflicts/ambiguities rather than preferring an attractive join."""
    by_accession = {r["version"]: r for r in refs}
    old = {r["source_id"]: r for r in candidates}
    voucher_by_unit, haul_by_unit = defaultdict(list), defaultdict(list)
    donors, fish_ids, photos = defaultdict(set), defaultdict(set), defaultdict(set)
    for ref in refs:
        if ref["voucher"] in metadata:
            donors[ref["version"]].add(ref["voucher"])
    for values in claims.values():
        for claim in values:
            row = claim["row"]
            if row["photo_id"]:
                photos[row["photo_id"]].add(row["specimen_id"])
    for claim in claims["voucher_crosswalk"]:
        row = claim["row"]
        donors[row["accession_version"]].add(row["specimen_id"])
        voucher_by_unit[row["specimen_id"]].append(claim)
    for claim in claims["haul_crosswalk"]:
        row = claim["row"]
        haul_by_unit[row["specimen_id"]].append(claim)
        if row["fish_id"]:
            fish_ids[(tuple(row[n] for n in HAUL_KEY), row["fish_id"])].add(row["specimen_id"])
    ledger = []
    for unit in sorted(metadata):
        vc, hc = voucher_by_unit[unit], haul_by_unit[unit]
        literal = [r for r in refs if r["voucher"] == unit]
        errors = []
        if any(len(donors[r["version"]]) > 1 for r in literal):
            errors.append("reference_reused_across_specimens")
        for claim in vc:
            row = claim["row"]
            if row["photo_id"] and len(photos[row["photo_id"]]) > 1:
                errors.append("photo_reused_across_specimens")
            ref = by_accession.get(row["accession_version"])
            if ref is None or ref["voucher"] != row["voucher"]:
                errors.append("accession_voucher_disagreement")
            if len(donors[row["accession_version"]]) != 1:
                errors.append("reference_reused_across_specimens")
            if literal and any(r["version"] != row["accession_version"] for r in literal):
                errors.append("literal_documentary_identity_conflict")
        vs = (
            "conflicting"
            if errors
            else "ambiguous"
            if len(vc) > 1 or len(literal) > 1
            else "source_documented"
            if vc
            else "literal_identity"
            if literal
            else "candidate"
            if old[unit]["accession_versions"]
            else "unresolved"
        )
        resolved = vs in ("source_documented", "literal_identity")
        versions = (
            ([vc[0]["row"]["accession_version"]] if vc else [r["version"] for r in literal])
            if resolved
            else []
        )
        he = []
        for claim in hc:
            key = tuple(claim["row"][n] for n in HAUL_KEY)
            row = claim["row"]
            if row["photo_id"] and len(photos[row["photo_id"]]) > 1:
                he.append("photo_reused_across_specimens")
            if row["fish_id"] and len(fish_ids[(key, row["fish_id"])]) > 1:
                he.append("sampling_number_reused_within_haul")
            if key not in catalogue:
                he.append("composite_haul_not_in_catalogue")
        hs = (
            "conflicting"
            if he
            else "ambiguous"
            if len(hc) > 1
            else "source_documented"
            if hc
            else "unresolved"
        )
        ledger.append(
            dict(
                specimen_id=unit,
                voucher_status=vs,
                accession_versions=versions,
                voucher_claims=vc,
                voucher_issues=sorted(set(errors)),
                candidate_reference=old[unit],
                haul_status=hs,
                haul_claims=hc,
                haul_issues=sorted(set(he)),
                haul_key=list(tuple(hc[0]["row"][n] for n in HAUL_KEY))
                if hs == "source_documented"
                else None,
                photo_identity="source_declared_only_not_independently_verified",
            )
        )
    outside = {
        role: [c for c in values if c["row"]["specimen_id"] not in metadata]
        for role, values in claims.items()
    }
    return ledger, outside


def coverage_status(count, total):
    return "complete" if count == total else "partial" if count else "unresolved"


def assess_correspondence(
    plan, *, prior_plan, source_dir, prior_audit_dir, evidence_dir, output_dir
):
    plan = CorrespondencePlan.from_dict(plan.to_dict())
    prior = ReplacementAuditPlan.from_dict(prior_plan.to_dict())
    metadata, refs, candidates, before = verify_prior(plan, prior, source_dir, prior_audit_dir)
    paths, evidence_before = verify_evidence(plan, evidence_dir)
    out = Path(output_dir).absolute()
    for root in (source_dir, prior_audit_dir, evidence_dir):
        root = Path(root).resolve()
        if out.resolve().is_relative_to(root) or root.is_relative_to(out.resolve()):
            raise ValueError("Assessment outputs must be separate from all inputs")
    out.mkdir(parents=True, exist_ok=False)
    try:
        write_json(plan.to_dict(), out / "specification.json")
        roles = defaultdict(list)
        for source in plan.sources:
            roles[source.role].append(paths[source.source_id])
        catalogue = haul_catalogue(roles["haul_catalogue"][0])
        dictionary = field_dictionary(roles["field_dictionary"][0])
        search = read_json(roles["search_register"][0])
        if (
            not isinstance(search, dict)
            or search.get("contact_performed") is not False
            or not isinstance(search.get("entries"), list)
            or not search["entries"]
        ):
            raise ValueError("Explicit bounded retrieval/search register without contact required")
        for entry in search["entries"]:
            if (
                not isinstance(entry, dict)
                or set(entry) != {"query_or_url", "outcome", "limitation"}
                or any(not isinstance(v, str) or not v.strip() for v in entry.values())
            ):
                raise ValueError("Every search outcome requires scope and limitation")
        claims = documentary_claims(plan, paths)
        ledger, outside = resolve_links(metadata, refs, candidates, claims, catalogue)
        voucher_n = sum(
            r["voucher_status"] in ("literal_identity", "source_documented") for r in ledger
        )
        haul_n = sum(r["haul_status"] == "source_documented" for r in ledger)
        used = {v for r in ledger for v in r["accession_versions"]}
        groups = defaultdict(list)
        for row in ledger:
            if row["haul_key"] is not None:
                groups[tuple(row["haul_key"])].append(row["specimen_id"])
        labels = {
            key: dict(sorted(Counter(metadata[u]["species"] for u in units).items()))
            for key, units in groups.items()
        }
        write_json(ledger, out / "specimen-correspondence.json")
        write_json(outside, out / "outside-cohort-claims.json")
        write_json(
            [dict(**r, linked_to_archive_specimen=r["version"] in used) for r in refs],
            out / "reference-accounting.json",
        )
        write_json(
            [
                dict(
                    haul_key=list(k),
                    source_record=v,
                    linked_specimens=groups.get(k, []),
                    source_label_counts=labels.get(k, {}),
                )
                for k, v in sorted(catalogue.items())
            ],
            out / "haul-accounting.json",
        )
        write_json(dictionary, out / "field-dictionary.json")
        write_json(search, out / "search-register.json")
        write_json(before, out / "prior-bindings.json")
        write_rows(out / "voucher-crosswalk.template.csv", VOUCHER_COLUMNS, [])
        write_rows(out / "haul-crosswalk.template.csv", HAUL_COLUMNS, [])
        write_rows(
            out / "records-needed.csv",
            ["specimen_id", "voucher_record_needed", "haul_record_needed"],
            [
                (
                    r["specimen_id"],
                    int(r["voucher_status"] not in ("literal_identity", "source_documented")),
                    int(r["haul_status"] != "source_documented"),
                )
                for r in ledger
            ],
        )
        write_json(
            dict(
                status="unpopulated_requirements_only",
                sent=False,
                contact_performed=False,
                voucher_fields=list(VOUCHER_COLUMNS),
                haul_fields=list(HAUL_COLUMNS),
                requirements=[
                    "Explicit specimen to complete voucher and versioned accession correspondence",
                    "Explicit specimen to original sampling number and composite "
                    "survey-haul correspondence",
                    "Provenance and corrections; photograph/capture replicates if applicable",
                ],
                templates_are_evidence=False,
                candidate_links_are_assignments=False,
            ),
            out / "records-needed.json",
        )
        summary = dict(
            schema_version=1,
            assessment_id=plan.assessment_id,
            phase=plan.phase,
            status="assessed",
            specification_sha256=configuration_hash(plan.to_dict()),
            metadata_records=len(metadata),
            reference_records=len(refs),
            survey_haul_records=len(catalogue),
            prior_candidate_specimens=sum(r["status"] == "candidate_namespace" for r in candidates),
            prior_unmatched_specimens=sum(r["status"] == "unmatched" for r in candidates),
            voucher_link_counts=dict(sorted(Counter(r["voucher_status"] for r in ledger).items())),
            haul_link_counts=dict(sorted(Counter(r["haul_status"] for r in ledger).items())),
            resolved_voucher_specimens=voucher_n,
            resolved_haul_specimens=haul_n,
            outside_cohort_claims={k: len(v) for k, v in outside.items()},
            references_without_documented_archive_link=len(refs) - len(used),
            linked_hauls=len(groups),
            mixed_source_label_hauls=sum(len(v) > 1 for v in labels.values()),
            source_label_haul_scope="resolved_links_only"
            if groups
            else "unavailable_no_specimen_haul_links",
            readiness=dict(
                voucher_correspondence=coverage_status(voucher_n, len(metadata)),
                haul_correspondence=coverage_status(haul_n, len(metadata)),
                empirical_analysis="not_authorized_no_frozen_analysis_design",
                biological_independence="not_established_by_mapping",
                independent_benchmark="not_established",
                significance="not_authorized",
            ),
            source_roles_are_reviewed_not_authenticated=True,
            database_mirrors_are_independent=False,
            contact_performed=False,
            email_sent=False,
            empirical_fits=0,
            pvalues_generated=0,
            stability_draws=0,
            significance_draws=0,
        )
        _, _, _, after = verify_prior(plan, prior, source_dir, prior_audit_dir)
        _, evidence_after = verify_evidence(plan, evidence_dir)
        if after != before or evidence_after != evidence_before:
            raise ValueError("Input evidence changed during correspondence assessment")
        pending = out / "summary.pending.json"
        write_json(summary, pending)
        hashes = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p != pending}
        hashes["summary.json"] = sha(pending)
        write_json(hashes, out / "checksums.json")
        pending.rename(out / "summary.json")
        return summary
    except Exception as exc:
        write_json(dict(status="failed", error=str(exc)), out / "failure.json")
        raise
