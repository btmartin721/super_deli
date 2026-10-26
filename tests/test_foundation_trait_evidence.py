"""Literal source links and fail-closed sampling/reference readiness."""

import copy
import gzip
import hashlib
import io
import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from zipfile import ZipFile

import pytest

from super_deli.cli import main
from super_deli.core.provenance import configuration_hash
from super_deli.core.trait_audit import TraitAuditPlan
from super_deli.core.trait_evidence import TraitEvidencePlan
from super_deli.io.config import read_json
from super_deli.io.evidence_workbook import _events, inspect_reference_workbook
from super_deli.io.trait_evidence import (
    assess_records,
    nexus_context,
    reference_tables,
    verify_source_inventories,
)
from super_deli.workflows.trait_audit import audit_trait_source
from super_deli.workflows.trait_evidence import assess_trait_evidence

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "examples/amphipod-gates/fixture"
PRIOR = ROOT / "examples/amphipod-audit/fixture"


def settings():
    return TraitEvidencePlan.from_dict(read_json(F / "plan.json"))


def ledger(path):
    return [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]


def world(tmp_path):
    prior = TraitAuditPlan.from_dict(read_json(PRIOR / "plan.json"))
    old = tmp_path / "prior"
    audit_trait_source(prior, source_dir=PRIOR, output_dir=old)
    evidence = tmp_path / "evidence"
    shutil.copytree(F, evidence)
    return settings(), prior, old, evidence


def run(tmp_path, state=None, name="assessment"):
    p, prior, old, evidence = state or world(tmp_path)
    return assess_trait_evidence(
        p,
        prior_plan=prior,
        source_dir=PRIOR,
        prior_audit_dir=old,
        evidence_dir=evidence,
        output_dir=tmp_path / name,
    )


def facts(tmp_path):
    p, prior, old, evidence = world(tmp_path)
    paths = {s.source_id: evidence / s.filename for s in p.sources}
    _, tables = reference_tables(paths)
    rows = ledger(old / "source-records.jsonl.gz")
    links = ledger(old / "reference-links.jsonl.gz")
    locations = ledger(old / "locality-links.jsonl.gz")
    context = nexus_context(paths["tree"], paths["regional"])
    return rows, links, locations, tables, context


def test_full_assessment_preserves_all_rows_and_unsupported_gates(tmp_path):
    result = run(tmp_path)
    assert result["source_rows"] == 6 and result["valid_stated_accessions"] == 4
    assert result["documented_specimen_links"] == result["molecular_partition_links"] == 2
    assert result["explicit_table_corroborations"] == 1
    assert result["correspondence_counts"]["conflicting_documentary_evidence"] == 1
    assert result["empty_formatting_rows"] == 1
    assert result["cohort_created"] is result["historical_partitions_replaced"] is False
    assert result["empirical_fits"] == result["bootstrap_draws"] == result["pvalues_generated"] == 0
    assert result["source_code_executed"] is result["contact_performed"] is False
    for use in (
        "independent_unit_fitting",
        "grouped_predictive_screen",
        "resampling",
        "significance",
        "species_boundary_validation",
    ):
        assert result["readiness"][use]["status"] == "blocked"
    for n, h in read_json(tmp_path / "assessment/checksums.json").items():
        assert hashlib.sha256((tmp_path / "assessment" / n).read_bytes()).hexdigest() == h


def test_immutable_separate_plan_roundtrip_and_prior_contract():
    p = settings()
    assert TraitEvidencePlan.from_dict(p.to_dict()) == p
    assert p.prior_specification_sha256 == configuration_hash(read_json(PRIOR / "plan.json"))
    with pytest.raises(FrozenInstanceError):
        p.phase = "fit"
    with pytest.raises(ValueError):
        TraitEvidencePlan.from_dict(dict(p.to_dict(), extra=True))


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("assessment_id", "../bad"),
        ("phase", "fit"),
        ("contact_policy", "email"),
        ("identity_policy", "guess"),
        ("reference_policy", "species_truth"),
        ("prior_checksums_sha256", "bad"),
        ("prior_specification_sha256", "bad"),
        ("sources", []),
        ("claims", []),
    ],
)
def test_plan_rejects_scope_or_binding_promotions(key, value):
    with pytest.raises(ValueError):
        TraitEvidencePlan.from_dict(dict(settings().to_dict(), **{key: value}))


@pytest.mark.parametrize(
    "key,value",
    [("status", "independent"), ("source_id", "absent"), ("anchor", ""), ("anchor", "x" * 161)],
)
def test_claims_cannot_promote_independence_or_lose_source_bindings(key, value):
    p = settings().to_dict()
    p["claims"][0][key] = value
    with pytest.raises(ValueError):
        TraitEvidencePlan.from_dict(p)


def test_absent_review_anchor_retains_failure_without_completed_summary(tmp_path):
    state = world(tmp_path)
    p = replace(
        state[0], claims=(replace(state[0].claims[0], anchor="absent text"), *state[0].claims[1:])
    )
    with pytest.raises(ValueError, match="anchor absent"):
        run(tmp_path, (p, *state[1:]))
    assert read_json(tmp_path / "assessment/failure.json")["status"] == "failed"
    assert not (tmp_path / "assessment/summary.json").exists()


def test_source_integrity_failure_is_retained(tmp_path):
    state = world(tmp_path)
    (state[3] / "trait.txt").write_text("changed")
    with pytest.raises(ValueError, match="size/hash"):
        run(tmp_path, state)
    assert (tmp_path / "assessment/failure.json").exists()
    assert not (tmp_path / "assessment/summary.json").exists()


def test_rewritten_prior_checksums_cannot_hide_altered_measurements(tmp_path):
    p, prior, old, evidence = world(tmp_path)
    target = old / "analysis/morphology.csv"
    target.write_text(target.read_text().replace("1.125", "9.125", 1))
    checks = read_json(old / "checksums.json")
    checks["analysis/morphology.csv"] = hashlib.sha256(target.read_bytes()).hexdigest()
    (old / "checksums.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
    p = replace(
        p, prior_checksums_sha256=hashlib.sha256((old / "checksums.json").read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="semantics differ"):
        run(tmp_path, (p, prior, old, evidence))
    assert not (tmp_path / "assessment/summary.json").exists()


def test_prior_specification_mismatch_is_not_silently_rebound(tmp_path):
    state = world(tmp_path)
    p = replace(state[0], prior_specification_sha256="a" * 64)
    with pytest.raises(ValueError, match="specification differs"):
        run(tmp_path, (p, *state[1:]))


def test_source_changes_at_completion_withhold_assessment(tmp_path, monkeypatch):
    import super_deli.workflows.trait_evidence as workflow

    state = world(tmp_path)
    original = workflow.readiness

    def changed(*args):
        (state[3] / "trait.txt").write_text("changed")
        return original(*args)

    monkeypatch.setattr(workflow, "readiness", changed)
    with pytest.raises(ValueError):
        run(tmp_path, state)
    assert (tmp_path / "assessment/failure.json").exists()
    assert not (tmp_path / "assessment/summary.json").exists()


def test_no_overwrite_or_outputs_within_evidence(tmp_path):
    state = world(tmp_path)
    run(tmp_path, state)
    with pytest.raises(FileExistsError):
        run(tmp_path, state)
    with pytest.raises(ValueError, match="separate"):
        assess_trait_evidence(
            state[0],
            prior_plan=state[1],
            source_dir=PRIOR,
            prior_audit_dir=state[2],
            evidence_dir=state[3],
            output_dir=state[3] / "out",
        )


def test_source_symlinks_are_rejected(tmp_path):
    state = world(tmp_path)
    p = state[3] / "trait.txt"
    p.unlink()
    p.symlink_to(F / "trait.txt")
    with pytest.raises(ValueError, match="hash mismatch"):
        run(tmp_path, state)


def test_streamed_empty_formatting_row_is_not_an_individual():
    b = inspect_reference_workbook(F / "reference01.xlsx")
    meta = b["worksheets"][0]
    assert meta["serialized_rows"] == 7 and meta["populated_rows"] == 6
    assert meta["empty_rows"] == 1 and meta["empty_row_ranges"] == [[10001, 10001]]
    assert len(b["sheets"]["Sheet1"]) == 6
    assert b["date1904"] == "0" and b["formula_evaluation"] is False
    assert b["external_link_following"] is False


@pytest.mark.parametrize(
    "raw",
    [b"<!DOCTYPE a><a/>", b'<!ENTITY a "b"><a/>', b"<broken>", "<a/>".encode("utf-16"), b"\xff"],
)
def test_streaming_xml_rejects_malformed_encodings_and_declarations(raw):
    with pytest.raises((ValueError, UnicodeError, ET.ParseError)):
        list(_events(io.BytesIO(raw)))


def test_entity_guard_covers_chunk_boundaries():
    raw = b" " * (64 * 1024 - 4) + b'<!DOCTYPE a [<!ENTITY x "unsafe">]><a>&x;</a>'
    with pytest.raises(ValueError, match="DTD/entities"):
        list(_events(io.BytesIO(raw)))


@pytest.mark.parametrize(
    "name", ["../bad", "/absolute", "bad\\name", "payload.exe", "payload.js", "xl/vbaProject.bin"]
)
def test_reference_zip_rejects_unsafe_members(tmp_path, name):
    p = tmp_path / "bad.xlsx"
    with ZipFile(p, "w") as z:
        z.writestr(name, b"not executed")
    with pytest.raises(ValueError):
        inspect_reference_workbook(p)


def test_published_inventory_is_checked_independently_of_snapshot_hashes(tmp_path):
    evidence = tmp_path / "evidence"
    shutil.copytree(F, evidence)
    paths = {s.source_id: evidence / s.filename for s in settings().sources}
    metadata = read_json(paths["reference_archive"])
    metadata["files"][0]["checksum"] = "md5:" + "0" * 32
    paths["reference_archive"].write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="inventory differs"):
        verify_source_inventories(paths)


def test_nexus_labels_are_context_and_never_a_specimen_crosswalk():
    context = nexus_context(F / "tree.txt", F / "regional.txt")
    assert context["tips"] == ["M1", "M2"] and context["regional_labels"] == ["M1"]
    assert context["tree_inferred"] is context["specimen_crosswalk"] is False
    assert nexus_context(None, None)["status"] == "unavailable"


@pytest.mark.parametrize(
    "value",
    [
        "#NEXUS Taxlabels M1 M1; Dimensions ntax=2;",
        "#NEXUS Taxlabels M1; Dimensions ntax=2;",
        '#NEXUS Taxlabels "M 1"; Dimensions ntax=1;',
    ],
)
def test_ambiguous_or_inferred_tree_labels_are_rejected(tmp_path, value):
    p = tmp_path / "tree"
    p.write_text(value)
    with pytest.raises(ValueError):
        nexus_context(p, None)


def test_new_table_cannot_override_a_conflicting_genbank_voucher(tmp_path):
    data = facts(tmp_path)
    out = assess_records(*data)
    row = next(x for x in out if x["voucher"] == "V04")
    assert row["previous_reference_status"] == "voucher_disagreement"
    assert row["genbank_conflict"] is True
    assert row["correspondence_status"] == "conflicting_documentary_evidence"
    assert row["documented_specimen_link"] is False


def test_explicit_pair_adds_evidence_without_inventing_genbank_qualifier(tmp_path):
    out = assess_records(*facts(tmp_path))
    row = next(x for x in out if x["voucher"] == "V06")
    assert row["previous_reference_status"] == "source_documented_record_voucher_absent"
    assert row["correspondence_status"] == "corroborated_explicit_table_pair"
    assert row["molecular_partition_link"] is True
    assert row["new_genbank_voucher_qualifier_asserted"] is False
    assert row["reference_species_truth_asserted"] is False


def test_matching_name_and_voucher_only_candidates_are_not_promoted(tmp_path):
    out = assess_records(*facts(tmp_path))
    row = next(x for x in out if x["voucher"] == "V05")
    assert row["voucher_only_candidates"] and row["sequence_name_only_candidates"]
    assert row["correspondence_status"] == "no_valid_stated_accession"
    assert row["documented_specimen_link"] is False
    assert row["molecular_partition_link"] is False


def test_source_flags_withhold_use_without_losing_raw_partitions(tmp_path):
    out = assess_records(*facts(tmp_path))
    row = next(x for x in out if x["voucher"] == "V02")
    assert row["correspondence_status"] == "withheld_source_caution"
    assert row["raw_molecular_partitions"] is not None and row["molecular_partition_link"] is False


@pytest.mark.parametrize("table", ["reference01", "reference02", "reference03"])
def test_duplicate_accession_rows_are_retained_as_ambiguity(tmp_path, table):
    data = facts(tmp_path)
    data[3][table].append(copy.deepcopy(data[3][table][0]))
    row = next(x for x in assess_records(*data) if x["voucher"] == "V01")
    assert row["correspondence_status"] == "ambiguous_documentary_accession"
    assert row["molecular_partition_link"] is False
    assert table in row["ambiguous_tables"]


def test_explicit_table_voucher_conflict_does_not_get_repaired_by_label(tmp_path):
    data = facts(tmp_path)
    data[3]["reference02"][0]["values"]["voucher"] = "OTHER"
    row = next(x for x in assess_records(*data) if x["voucher"] == "V01")
    assert row["correspondence_status"] == "conflicting_documentary_evidence"
    assert row["contradictory_voucher_records"]


def test_composite_record_cannot_bridge_a_missing_genbank_voucher(tmp_path):
    data = facts(tmp_path)
    for r in data[3]["reference02"]:
        if r["values"]["voucher"] == "V06":
            r["values"]["chimera"] = "Other specimen (COI)"
    row = next(x for x in assess_records(*data) if x["voucher"] == "V06")
    assert row["composite_multilocus_records"]
    assert row["correspondence_status"] == "source_asserted_accession_only"
    assert row["molecular_partition_link"] is False


def test_different_partition_label_strings_do_not_imply_disagreement(tmp_path):
    out = assess_records(*facts(tmp_path))
    row = next(x for x in out if x["voucher"] == "V06")
    assert row["raw_molecular_partitions"] == {"ptp": "P3", "asap": "A4"}
    assert row["partition_relation"] == "one_to_one"
    split = next(x for x in out if x["voucher"] == "V01")
    assert split["partition_relation"] == "split_or_merge_context"


def test_label_and_coordinate_changes_cannot_determine_specimen_correspondence(tmp_path):
    data = facts(tmp_path)
    before = assess_records(*data)
    for r in data[0]:
        for n in ("MOTU", "species", "locality_name", "latitude", "longitude"):
            r["values"][n] = "changed metadata"
    for r in data[3]["reference01"]:
        for n in (
            "PTP species delimitation",
            "ASAP species delimitation",
            "MOTU_SubBioLab",
            "Latitude",
            "Longitude",
            "Locality name",
        ):
            r["values"][n] = "changed metadata"
    after = assess_records(*data)
    assert [r["correspondence_status"] for r in before] == [
        r["correspondence_status"] for r in after
    ]
    assert all(r["independent_sampling_unit_asserted"] is False for r in after)


def test_formulas_in_partition_keys_are_not_admitted(tmp_path):
    data = facts(tmp_path)
    data[3]["reference01"][0]["cells"]["PTP species delimitation"]["formula"] = {"text": "1+1"}
    row = next(x for x in assess_records(*data) if x["voucher"] == "V01")
    assert row["documented_specimen_link"] is True and row["molecular_partition_link"] is False


def test_absent_reference_tables_cannot_create_links(tmp_path):
    data = facts(tmp_path)
    for rows in data[3].values():
        rows.clear()
    out = assess_records(*data)
    assert not any(x["molecular_partition_link"] for x in out)
    assert next(x for x in out if x["voucher"] == "V06")["documented_specimen_link"] is False


def test_whole_workflow_replay_is_byte_identical(tmp_path):
    state = world(tmp_path)
    run(tmp_path, state, name="a")
    run(tmp_path, state, name="b")
    for p in (tmp_path / "a").rglob("*"):
        if p.is_file():
            assert p.read_bytes() == (tmp_path / "b" / p.relative_to(tmp_path / "a")).read_bytes()


def test_cli_has_no_fitting_permission(tmp_path, capsys):
    p, prior, old, evidence = world(tmp_path)
    assert (
        main(
            [
                "assess-trait-evidence",
                str(F / "plan.json"),
                "--prior-plan",
                str(PRIOR / "plan.json"),
                "--source-dir",
                str(PRIOR),
                "--prior-audit-dir",
                str(old),
                "--evidence-dir",
                str(evidence),
                "--output-dir",
                str(tmp_path / "cli"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["empirical_fits"] == 0


def test_date_system_row_attributes_and_merged_blanks_are_preserved(tmp_path):
    source = tmp_path / "reference.xlsx"
    with ZipFile(F / "reference01.xlsx") as z:
        members = [(i, z.read(i.filename)) for i in z.infolist()]
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with ZipFile(source, "w") as z:
        for info, raw in members:
            if info.filename == "xl/workbook.xml":
                root = ET.fromstring(raw)
                root.find(namespace + "workbookPr").set("date1904", "1")
                raw = ET.tostring(root)
            elif info.filename == "xl/worksheets/sheet1.xml":
                root = ET.fromstring(raw)
                root.find(namespace + "sheetData")[1].set("hidden", "1")
                cell = next(c for c in root.find(namespace + "sheetData")[2] if c.get("r") == "A3")
                cell.find(namespace + "is/" + namespace + "t").text = ""
                merges = ET.SubElement(root, namespace + "mergeCells")
                ET.SubElement(merges, namespace + "mergeCell", ref="A2:A3")
                raw = ET.tostring(root)
            z.writestr(info, raw)
    book = inspect_reference_workbook(source)
    assert book["date1904"] == "1"
    assert book["sheets"]["Sheet1"][1]["attributes"]["hidden"] == "1"
    assert book["sheets"]["Sheet1"][2]["cells"]["A"]["value"] == ""
    assert book["worksheets"][0]["merged_ranges"] == ["A2:A3"]


def test_malformed_archive_inventory_has_a_controlled_error(tmp_path):
    p = tmp_path / "archive.json"
    p.write_text('{"files": [null]}')
    with pytest.raises(ValueError, match="inventory required"):
        verify_source_inventories({"reference_archive": p})


def test_malformed_context_inventory_has_a_controlled_error(tmp_path):
    p = tmp_path / "meta.json"
    p.write_text('{"files": [1]}')
    with pytest.raises(ValueError, match="inventory differs"):
        verify_source_inventories({"tree_meta": p, "tree": F / "tree.txt"})


def test_explicit_version_conflict_withholds_documentary_use(tmp_path):
    data = facts(tmp_path)
    data[0][0]["values"]["COI_sequence"] = "XY000000.2"
    data[3]["reference01"][0]["values"]["GenBank accesion number"] = "XY000000.1"
    row = assess_records(*data)[0]
    assert row["version_conflicts"]
    assert row["correspondence_status"] == "conflicting_documentary_evidence"
    assert row["molecular_partition_link"] is False
