"""Documentary identity evidence, collision handling and fit-free boundaries."""

import csv
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from super_deli.cli import main
from super_deli.core.correspondence import CorrespondencePlan
from super_deli.core.replacement_audit import ReplacementAuditPlan
from super_deli.io.config import read_json
from super_deli.io.correspondence import HAUL_KEY, crosswalk_rows, field_dictionary, haul_catalogue
from super_deli.io.specimen_audit import genbank_references, source_tables
from super_deli.workflows.correspondence import assess_correspondence, resolve_links
from super_deli.workflows.replacement_audit import (
    audit_replacement,
    reference_links,
    verify_sources,
)

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "examples/skate-mappings/fixture"
PRIOR = ROOT / "examples/replacement-empirical/fixture"


def settings():
    return CorrespondencePlan.from_dict(read_json(F / "plan.json"))


def world(tmp_path):
    prior = ReplacementAuditPlan.from_dict(read_json(PRIOR / "plan.json"))
    audit = tmp_path / "prior"
    audit_replacement(prior, source_dir=PRIOR, output_dir=audit)
    evidence = tmp_path / "evidence"
    shutil.copytree(F, evidence)
    p = settings()
    assert (
        p.prior_checksums_sha256
        == hashlib.sha256((audit / "checksums.json").read_bytes()).hexdigest()
    )
    return p, prior, audit, evidence


def run(tmp_path, *, state=None, plan=None, name="assessment"):
    p, prior, audit, evidence = state or world(tmp_path)
    return assess_correspondence(
        plan or p,
        prior_plan=prior,
        source_dir=PRIOR,
        prior_audit_dir=audit,
        evidence_dir=evidence,
        output_dir=tmp_path / name,
    )


def edit(state, filename, mutate):
    p, prior, audit, evidence = state
    path = evidence / filename
    mutate(path)
    p = replace(
        p,
        sources=tuple(
            replace(
                s,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size_bytes=path.stat().st_size,
            )
            if s.filename == filename
            else s
            for s in p.sources
        ),
    )
    return p, prior, audit, evidence


def table_edit(path, mutate):
    with path.open(newline="") as h:
        reader = csv.DictReader(h)
        columns = reader.fieldnames
        rows = list(reader)
    mutate(rows)
    with path.open("w", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def ledger(tmp_path):
    return {
        r["specimen_id"]: r for r in read_json(tmp_path / "assessment/specimen-correspondence.json")
    }


def test_complete_documentary_links_remain_separate_from_analysis(tmp_path):
    result = run(tmp_path)
    assert result["resolved_voucher_specimens"] == result["resolved_haul_specimens"] == 4
    assert result["readiness"]["voucher_correspondence"] == "complete"
    assert result["readiness"]["haul_correspondence"] == "complete"
    assert result["readiness"]["empirical_analysis"].startswith("not_authorized")
    assert result["readiness"]["independent_benchmark"] == "not_established"
    assert result["linked_hauls"] == 3 and result["mixed_source_label_hauls"] == 1
    assert result["empirical_fits"] == result["pvalues_generated"] == 0
    assert result["contact_performed"] is False and result["email_sent"] is False
    assert ledger(tmp_path)["U03"]["accession_versions"] == ["XY000003.1"]
    assert read_json(tmp_path / "assessment/records-needed.json")["templates_are_evidence"] is False


def test_absent_crosswalks_do_not_promote_unique_candidates(tmp_path):
    state = world(tmp_path)
    p = replace(
        state[0],
        sources=tuple(
            s for s in state[0].sources if s.role not in {"voucher_crosswalk", "haul_crosswalk"}
        ),
    )
    result = run(tmp_path, state=state, plan=p)
    assert result["voucher_link_counts"] == {"candidate": 3, "unresolved": 1}
    assert result["prior_candidate_specimens"] == 3
    assert result["prior_unmatched_specimens"] == 1
    assert result["haul_link_counts"] == {"unresolved": 4}
    assert result["linked_hauls"] == 0
    assert result["source_label_haul_scope"] == "unavailable_no_specimen_haul_links"
    assert result["resolved_voucher_specimens"] == result["resolved_haul_specimens"] == 0
    assert len(read_json(tmp_path / "assessment/reference-accounting.json")) == 4
    assert len(read_json(tmp_path / "assessment/haul-accounting.json")) == 3


def test_partial_and_outside_cohort_links_keep_full_denominators(tmp_path):
    state = edit(
        world(tmp_path),
        "vouchers.csv",
        lambda p: table_edit(
            p, lambda r: r.__setitem__(slice(1, None), [dict(r[1], specimen_id="OUTSIDE")])
        ),
    )
    result = run(tmp_path, state=state)
    assert result["metadata_records"] == 4 and result["resolved_voucher_specimens"] == 1
    assert result["readiness"]["voucher_correspondence"] == "partial"
    assert result["outside_cohort_claims"]["voucher_crosswalk"] == 1
    assert len(read_json(tmp_path / "assessment/specimen-correspondence.json")) == 4


@pytest.mark.parametrize("filename", ["vouchers.csv", "sampling.csv"])
def test_duplicate_documentary_rows_are_ambiguous(tmp_path, filename):
    state = edit(world(tmp_path), filename, lambda p: table_edit(p, lambda r: r.append(dict(r[0]))))
    run(tmp_path, state=state)
    assert (
        ledger(tmp_path)["U01"]["voucher_status" if filename == "vouchers.csv" else "haul_status"]
        == "ambiguous"
    )


@pytest.mark.parametrize(
    "column,value",
    [
        ("accession_version", "XY999999.1"),
        ("voucher", "LAB-A-U001"),
        ("accession_version", "XY000000.2"),
    ],
)
def test_mismatched_or_undeclared_alias_targets_remain_conflicting(tmp_path, column, value):
    state = edit(
        world(tmp_path),
        "vouchers.csv",
        lambda p: table_edit(p, lambda r: r[0].__setitem__(column, value)),
    )
    run(tmp_path, state=state)
    assert ledger(tmp_path)["U01"]["voucher_status"] == "conflicting"
    assert ledger(tmp_path)["U01"]["accession_versions"] == []


def test_reference_reuse_across_specimens_is_not_resolved(tmp_path):
    def mutate(rows):
        rows[1].update(voucher=rows[0]["voucher"], accession_version=rows[0]["accession_version"])

    state = edit(world(tmp_path), "vouchers.csv", lambda p: table_edit(p, mutate))
    run(tmp_path, state=state)
    assert all(ledger(tmp_path)[u]["voucher_status"] == "conflicting" for u in ("U01", "U02"))


def test_photo_reuse_preserves_each_mapping_scope(tmp_path):
    def mutate(rows):
        rows[1]["photo_id"] = rows[0]["photo_id"]

    state = edit(world(tmp_path), "vouchers.csv", lambda p: table_edit(p, mutate))
    run(tmp_path, state=state)
    assert all(ledger(tmp_path)[u]["voucher_status"] == "conflicting" for u in ("U01", "U02"))
    assert ledger(tmp_path)["U01"]["haul_status"] == "conflicting"
    assert ledger(tmp_path)["U02"]["haul_status"] == "source_documented"


def test_running_sampling_number_is_scoped_to_full_haul_identity(tmp_path):
    state = edit(
        world(tmp_path),
        "sampling.csv",
        lambda p: table_edit(p, lambda r: r[1].__setitem__("fish_id", r[0]["fish_id"])),
    )
    run(tmp_path, state=state)
    assert all(ledger(tmp_path)[u]["haul_status"] == "conflicting" for u in ("U01", "U02"))
    assert all(ledger(tmp_path)[u]["haul_status"] == "source_documented" for u in ("U03", "U04"))


def test_unknown_composite_haul_is_retained_as_conflict(tmp_path):
    state = edit(
        world(tmp_path),
        "sampling.csv",
        lambda p: table_edit(p, lambda r: r[0].__setitem__("Ship", "OTHER")),
    )
    run(tmp_path, state=state)
    assert ledger(tmp_path)["U01"]["haul_issues"] == ["composite_haul_not_in_catalogue"]


def test_short_id_padding_is_never_inferred(tmp_path):
    state = edit(
        world(tmp_path),
        "vouchers.csv",
        lambda p: table_edit(p, lambda r: r[0].__setitem__("specimen_id", "U1")),
    )
    result = run(tmp_path, state=state)
    assert ledger(tmp_path)["U01"]["voucher_status"] == "candidate"
    assert result["outside_cohort_claims"]["voucher_crosswalk"] == 1


def test_empty_templates_are_not_completed_evidence(tmp_path):
    state = world(tmp_path)
    for name in ("vouchers.csv", "sampling.csv"):
        state = edit(state, name, lambda p: table_edit(p, lambda r: r.clear()))
    result = run(tmp_path, state=state)
    assert result["resolved_voucher_specimens"] == result["resolved_haul_specimens"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"phase": "fit"},
        {"identity_policy": "strip_zeroes"},
        {"contact_policy": "email_authors"},
        {"prior_checksums_sha256": "x"},
        {"assessment_id": "Person Name"},
    ],
)
def test_plan_rejects_unsupported_promotions(kwargs):
    with pytest.raises(ValueError):
        replace(settings(), **kwargs)


def test_strict_roundtrip_and_unknown_fields():
    p = settings()
    assert CorrespondencePlan.from_dict(p.to_dict()) == p
    for changed in (dict(p.to_dict(), authorize_fit=True), dict(p.to_dict(), sources="none")):
        with pytest.raises(ValueError):
            CorrespondencePlan.from_dict(changed)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"filename": "../source.csv"},
        {"filename": "source\x00.csv"},
        {"url": "http://example.org/x"},
        {"url": "https://user:secret@example.org/x"},
        {"sha256": "0"},
        {"size_bytes": True},
        {"size_bytes": 8 * 1024**2 + 1},
        {"role": "normalization_rule"},
    ],
)
def test_source_declaration_rejects_invalid_evidence(kwargs):
    with pytest.raises(ValueError):
        replace(settings().sources[0], **kwargs)


def test_missing_or_duplicate_required_evidence_roles():
    p = settings()
    with pytest.raises(ValueError):
        replace(p, sources=p.sources[1:])
    with pytest.raises(ValueError):
        replace(p, sources=(*p.sources, p.sources[0]))


def test_documentary_tables_reject_undeclared_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("specimen_id,voucher,accession_version,photo_id,inferred\nU01,V01,XY1.1,,yes\n")
    with pytest.raises(ValueError):
        crosswalk_rows(p, "voucher_crosswalk")


def test_whitespace_padding_is_not_a_normalization_rule(tmp_path):
    state = edit(
        world(tmp_path),
        "vouchers.csv",
        lambda p: table_edit(p, lambda r: r[0].__setitem__("specimen_id", " U01")),
    )
    with pytest.raises(ValueError):
        run(tmp_path, state=state)
    assert (tmp_path / "assessment/failure.json").exists()
    assert not (tmp_path / "assessment/summary.json").exists()


def test_composite_catalogue_retains_repeated_haul_numbers():
    catalogue = haul_catalogue(F / "hauls.xml")
    assert len(catalogue) == 3 and sum(k[-1] == "1" for k in catalogue) == 2
    assert all(len(k) == len(HAUL_KEY) for k in catalogue)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda a: a.replace("TEST2", "TEST1").replace("<StNo>B</StNo>", "<StNo>A</StNo>"),
        lambda a: a.replace("<Year>2021</Year>", "<Year>2020</Year>"),
        lambda a: a.replace("<Survey>IE-IGFS</Survey>", "<Survey>OTHER</Survey>"),
        lambda a: a.replace("<HaulNo>1</HaulNo>", ""),
        lambda a: '<!DOCTYPE x [<!ENTITY evil "x">]>' + a,
    ],
)
def test_bad_or_out_of_scope_haul_catalogues_are_rejected(tmp_path, mutation):
    p = tmp_path / "hauls.xml"
    p.write_text(mutation((F / "hauls.xml").read_text()))
    with pytest.raises((ValueError, SyntaxError)):
        haul_catalogue(p)


def test_field_dictionary_requires_official_structure(tmp_path):
    assert field_dictionary(F / "fields.xml")[0]["FieldName"] == "FishID"
    p = tmp_path / "bad.xml"
    p.write_text("<fields/>")
    with pytest.raises(ValueError):
        field_dictionary(p)


def test_evidence_mutation_and_symlink_do_not_create_an_assessment(tmp_path):
    state = world(tmp_path)
    p = state[3] / "vouchers.csv"
    p.write_text(p.read_text() + "\n")
    with pytest.raises(ValueError):
        run(tmp_path, state=state)
    assert not (tmp_path / "assessment").exists()
    p.unlink()
    p.symlink_to(F / "vouchers.csv")
    with pytest.raises(ValueError):
        run(tmp_path, state=state)


def test_prior_inventory_and_bound_content_are_checked(tmp_path):
    state = world(tmp_path)
    p = state[2] / "reference-links.json"
    p.write_text("[]")
    with pytest.raises(ValueError):
        run(tmp_path, state=state)
    assert not (tmp_path / "assessment").exists()


def test_output_overlap_and_overwrite_are_rejected(tmp_path):
    state = world(tmp_path)
    p, prior, audit, evidence = state
    for target in (evidence / "nested", audit / "nested", evidence.parent):
        with pytest.raises((ValueError, FileExistsError)):
            assess_correspondence(
                p,
                prior_plan=prior,
                source_dir=PRIOR,
                prior_audit_dir=audit,
                evidence_dir=evidence,
                output_dir=target,
            )
    run(tmp_path, state=state)
    with pytest.raises(FileExistsError):
        run(tmp_path, state=state)


def test_failed_search_register_retains_failure_without_completed_summary(tmp_path):
    state = edit(
        world(tmp_path),
        "search.json",
        lambda p: p.write_text(json.dumps(dict(contact_performed=True, entries=[]))),
    )
    with pytest.raises(ValueError):
        run(tmp_path, state=state)
    assert read_json(tmp_path / "assessment/failure.json")["status"] == "failed"
    assert not (tmp_path / "assessment/summary.json").exists()


def test_late_input_mutation_withholds_summary(tmp_path, monkeypatch):
    import super_deli.workflows.correspondence as module

    state = world(tmp_path)
    original = module.documentary_claims

    def changed(*args):
        claims = original(*args)
        p = state[3] / "search.json"
        p.write_text(p.read_text() + "\n")
        return claims

    monkeypatch.setattr(module, "documentary_claims", changed)
    with pytest.raises(ValueError):
        run(tmp_path, state=state)
    assert not (tmp_path / "assessment/summary.json").exists()
    assert (tmp_path / "assessment/failure.json").exists()


def test_deterministic_outputs(tmp_path):
    state = world(tmp_path)
    run(tmp_path, state=state, name="one")
    run(tmp_path, state=state, name="two")
    assert {p.name: p.read_bytes() for p in (tmp_path / "one").iterdir()} == {
        p.name: p.read_bytes() for p in (tmp_path / "two").iterdir()
    }


def test_row_order_and_source_labels_do_not_choose_links(tmp_path):
    import super_deli.workflows.correspondence as module

    state = world(tmp_path)
    p, prior, audit, evidence = state
    paths, _ = verify_sources(prior, PRIOR)
    metadata, _ = source_tables(paths["metadata"], paths["colour"])
    refs = genbank_references(paths["references"])
    candidates = reference_links(metadata, refs, prior)
    ep, _ = module.verify_evidence(p, evidence)
    claims = module.documentary_claims(p, ep)
    catalogue = haul_catalogue(F / "hauls.xml")
    first, _ = resolve_links(metadata, refs, candidates, claims, catalogue)
    metadata = {
        u: dict(row, species="beta" if row["species"] == "alpha" else "alpha")
        for u, row in reversed(list(metadata.items()))
    }
    candidates = reference_links(metadata, list(reversed(refs)), prior)
    second, _ = resolve_links(
        metadata,
        list(reversed(refs)),
        candidates,
        {k: list(reversed(v)) for k, v in claims.items()},
        catalogue,
    )
    for key in ("specimen_id", "voucher_status", "accession_versions", "haul_status", "haul_key"):
        assert [r[key] for r in first] == [r[key] for r in second]


def test_installed_cli_contract(tmp_path, capsys):
    p, prior, audit, evidence = world(tmp_path)
    assert (
        main(
            [
                "assess-correspondence",
                str(F / "plan.json"),
                "--prior-plan",
                str(PRIOR / "plan.json"),
                "--source-dir",
                str(PRIOR),
                "--prior-audit-dir",
                str(audit),
                "--evidence-dir",
                str(evidence),
                "--output-dir",
                str(tmp_path / "assessment"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["resolved_voucher_specimens"] == 4


def test_official_fixed_width_padding_is_preserved_without_normalization(tmp_path):
    state = world(tmp_path)
    state = edit(
        state,
        "hauls.xml",
        lambda p: p.write_text(
            p.read_text().replace("<Country>SYN</Country>", "<Country>SYN  </Country>")
        ),
    )
    result = run(tmp_path, state=state, name="unmatched_padding")
    assert result["resolved_haul_specimens"] == 0
    state = edit(
        state,
        "sampling.csv",
        lambda p: table_edit(p, lambda rows: [r.__setitem__("Country", "SYN  ") for r in rows]),
    )
    result = run(tmp_path, state=state)
    assert result["resolved_haul_specimens"] == 4
    assert all(r["haul_key"][3] == "SYN  " for r in ledger(tmp_path).values())
