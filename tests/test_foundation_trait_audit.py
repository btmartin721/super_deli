"""Fit-free provenance, literal identity, missingness and source-safety boundaries."""

import copy
import csv
import gzip
import hashlib
import json
import shutil
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from zipfile import ZipFile

import pytest

from super_deli.cli import main
from super_deli.core.trait_audit import COLUMNS, TRAITS, TraitAuditPlan
from super_deli.io.audit_workbook import S, _sheet, inspect_workbook, xml_bytes
from super_deli.io.config import read_json
from super_deli.io.trait_source import (
    inspect_rows,
    locality_accounting,
    molecular_records,
    numeric_cell,
    reference_accounting,
    source_rows,
    trait_profiles,
)
from super_deli.workflows.trait_audit import (
    audit_trait_source,
    export_representation,
    verify_trait_sources,
    write_gz_rows,
)

F = Path(__file__).resolve().parents[1] / "examples/amphipod-audit/fixture"


def plan():
    return TraitAuditPlan.from_dict(read_json(F / "plan.json"))


def records():
    b = inspect_workbook(F / "traits.xlsx")
    rows, _, _ = source_rows(b)
    inspect_rows(rows)
    return rows, b


def run(tmp_path, p=None, sources=F, name="audit"):
    return audit_trait_source(p or plan(), source_dir=sources, output_dir=tmp_path / name)


def test_complete_workflow_keeps_evidence_roles_and_gates_separate(tmp_path):
    a = run(tmp_path)
    assert a["source_rows"] == 6 and a["source_columns"] == 45
    assert a["identity_counts"] == {"unique": 5, "missing": 1}
    assert a["reference_link_counts"] == {
        "literal_voucher_coi_match": 2,
        "not_stated": 1,
        "placeholder": 1,
        "source_documented_record_voucher_absent": 1,
        "voucher_disagreement": 1,
    }
    assert a["requested_accessions"] == a["returned_reference_records"] == 4
    assert a["unflagged_literal_links"] == 1
    assert a["formula_cells"] == 1 and a["formula_evaluation"] is False
    assert a["representation"]["observations"] == 5
    assert a["representation"]["statistical_cohort_authorized"] is False
    assert a["empirical_fits"] == a["pvalues_generated"] == a["bootstrap_draws"] == 0
    assert a["contact_performed"] is a["email_sent"] is False
    assert a["readiness"]["reference_label_independence"].startswith("not_established")
    assert a["readiness"]["biological_independence"] == "requires_separate_sampling_design"
    for n, h in read_json(tmp_path / "audit/checksums.json").items():
        assert hashlib.sha256((tmp_path / "audit" / n).read_bytes()).hexdigest() == h


def test_strict_immutable_plan_roundtrip():
    p = plan()
    assert TraitAuditPlan.from_dict(p.to_dict()) == p
    with pytest.raises(FrozenInstanceError):
        p.phase = "fit"
    with pytest.raises(ValueError):
        TraitAuditPlan.from_dict(dict(p.to_dict(), extra=True))


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("audit_id", "../bad"),
        ("phase", "clustering"),
        ("profile_id", "generic"),
        ("contact_policy", "email"),
        ("reference_policy", "species_truth"),
        ("identity_policy", "guess_aliases"),
        ("sources", []),
        ("published_counts", [["rows", True], ["motu_labels", 2], ["locality_names", 2]]),
        ("published_counts", [["rows", 1], ["rows", 2], ["locality_names", 2]]),
    ],
)
def test_plan_rejects_unsupported_scope(key, value):
    with pytest.raises(ValueError):
        TraitAuditPlan.from_dict(dict(plan().to_dict(), **{key: value}))


@pytest.mark.parametrize(
    "key,value",
    [
        ("role", "unknown"),
        ("filename", "../file"),
        ("filename", " file"),
        ("url", "http://example.org"),
        ("url", "https://u:p@example.org"),
        ("size_bytes", True),
        ("size_bytes", 9 * 1024**2),
        ("sha256", "no"),
        ("access", "assumed"),
        ("access", "unavailable"),
    ],
)
def test_snapshot_rejects_unsafe_or_unlocked_source(key, value):
    with pytest.raises(ValueError):
        replace(plan().sources[0], **{key: value})


def test_unavailable_reference_retains_full_denominators(tmp_path):
    p = replace(
        plan(),
        sources=tuple(
            replace(s, access="unavailable", size_bytes=0, sha256=None)
            if s.role in {"references", "reference_text"}
            else s
            for s in plan().sources
        ),
    )
    src = tmp_path / "sources"
    shutil.copytree(F, src)
    for n in ("references.xml", "references.gb"):
        (src / n).unlink()
    a = run(tmp_path, p, src)
    assert a["returned_reference_records"] == 0
    assert a["missing_requested_accessions"] == 4
    assert a["reference_link_counts"]["retrieval_unavailable"] == 4
    assert a["unflagged_literal_links"] == 0


def test_sources_are_hash_locked_and_symlinks_are_rejected(tmp_path):
    src = tmp_path / "sources"
    shutil.copytree(F, src)
    (src / "publication.xml").write_text("changed")
    with pytest.raises(ValueError, match="size differs"):
        verify_trait_sources(plan(), src)
    (src / "publication.xml").unlink()
    (src / "publication.xml").symlink_to(F / "publication.xml")
    with pytest.raises(ValueError, match="symbolic"):
        verify_trait_sources(plan(), src)


def test_outputs_do_not_overwrite_or_mix_with_sources(tmp_path):
    run(tmp_path)
    with pytest.raises(FileExistsError):
        run(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        audit_trait_source(plan(), source_dir=F, output_dir=F / "out")


def test_failure_has_no_success_summary(tmp_path, monkeypatch):
    import super_deli.workflows.trait_audit as workflow

    def broken(*args):
        raise ValueError("retained failure")

    monkeypatch.setattr(workflow, "inspect_workbook", broken)
    with pytest.raises(ValueError, match="retained failure"):
        run(tmp_path)
    assert read_json(tmp_path / "audit/failure.json")["status"] == "failed"
    assert not (tmp_path / "audit/summary.json").exists()


def test_source_mutation_during_audit_withholds_success(tmp_path, monkeypatch):
    import super_deli.workflows.trait_audit as workflow

    src = tmp_path / "sources"
    shutil.copytree(F, src)
    original = workflow.trait_profiles

    def changed(rows):
        (src / "publication.xml").write_text("changed")
        return original(rows)

    monkeypatch.setattr(workflow, "trait_profiles", changed)
    with pytest.raises(ValueError, match="size differs"):
        run(tmp_path, sources=src)
    assert (tmp_path / "audit/failure.json").exists()
    assert not (tmp_path / "audit/summary.json").exists()


@pytest.mark.parametrize(
    "raw", [b"<!DOCTYPE a><a/>", b'<!ENTITY e "x"><a/>', b"<a>" + b" " * (16 * 1024**2) + b"</a>"]
)
def test_unsafe_or_unbounded_xml_rejected(raw):
    with pytest.raises(ValueError):
        xml_bytes(raw)


@pytest.mark.parametrize(
    "name",
    ["../outside", "/absolute", "bad\\name", "xl/vbaProject.bin", "payload.js", "payload.exe"],
)
def test_unsafe_archive_payload_rejected(tmp_path, name):
    p = tmp_path / "bad.xlsx"
    with ZipFile(p, "w") as z:
        z.writestr(name, b"not executed")
    with pytest.raises(ValueError):
        inspect_workbook(p)


@pytest.mark.parametrize(
    "cells",
    [
        '<c r="A2"><v>1</v></c>',
        '<c r="A1"/><c r="A1"/>',
        '<c r="A1" t="s"><v>3</v></c>',
        '<c r="A1" t="b"><v>2</v></c>',
        '<c r="A1"><v>1</v><v>2</v></c>',
        '<c r="A1" t="bad"/>',
    ],
)
def test_cell_coordinate_type_and_cache_ambiguity_rejected(cells):
    raw = (
        f'<worksheet xmlns="{S[1:-1]}"><sheetData><row r="1">{cells}</row></sheetData></worksheet>'
    )
    with pytest.raises(ValueError):
        _sheet(raw.encode(), [])


@pytest.mark.parametrize(
    "value,kind,expected",
    [
        ("NA", "s", None),
        ("", "n", None),
        ("unknown", "s", None),
        ("-2.5", "n", -2.5),
        ("0", "n", 0),
        ("1e-3", "str", 0.001),
    ],
)
def test_numeric_tokens_preserved_without_imputation(value, kind, expected):
    assert numeric_cell(dict(value=value, type=kind)) == (expected, None)


@pytest.mark.parametrize(
    "value,kind",
    [
        ("NaN", "n"),
        ("inf", "n"),
        ("1,2?", "s"),
        (" 2", "n"),
        ("1", "b"),
        ("2020-01-01", "d"),
        ("#REF!", "e"),
    ],
)
def test_invalid_measurements_retained_as_issues(value, kind):
    assert numeric_cell(dict(value=value, type=kind))[1]


def test_formula_and_excel_date_are_not_evaluated_or_converted():
    rows, b = records()
    row = rows[4]
    assert row["values"]["body_l"] == "5.0"
    assert row["cells"]["body_l"]["formula"]["text"] == "2+3"
    cell = dict(row["cells"]["body_l"], cache_present=False)
    assert numeric_cell(cell) == (None, "formula_without_cache")
    links, _, _ = locality_accounting(rows, b)
    assert links[0]["survey_date"] == "45000"
    assert links[0]["date_style"] == "1"
    assert links[0]["date_kind"] == "stored_numeric_not_converted"
    assert links[1]["date_kind"] == "year_range"
    assert all(x["statistical_sampling_group_asserted"] is False for x in links)


def test_duplicate_and_missing_vouchers_never_gain_technical_biological_ids():
    rows, _ = records()
    rows[1]["values"]["voucher"] = rows[0]["values"]["voucher"]
    inspect_rows(rows)
    assert rows[0]["identity_status"] == rows[1]["identity_status"] == "duplicate"
    assert rows[0]["unit_id"] is rows[1]["unit_id"] is rows[2]["unit_id"] is None
    assert all(not r["representation_eligible"] for r in rows[:3])


@pytest.mark.parametrize(
    "mutate,status",
    [
        ("missing", "not_stated"),
        ("placeholder", "placeholder"),
        ("malformed", "malformed_accession"),
        ("absent", "not_returned"),
        ("version", "version_disagreement"),
        ("wrong_voucher", "voucher_disagreement"),
        ("no_voucher", "source_documented_record_voucher_absent"),
        ("wrong_marker", "marker_not_coi"),
        ("identity", "source_voucher_unresolved"),
        ("duplicate", "ambiguous_accession_usage"),
    ],
)
def test_reference_link_failure_reasons_are_not_promoted(mutate, status):
    rows, _ = records()
    refs = molecular_records(F / "references.xml")
    if mutate in {"missing", "placeholder", "malformed", "absent", "version"}:
        rows[0]["values"]["COI_sequence"] = {
            "missing": "NA",
            "placeholder": "check",
            "malformed": "123",
            "absent": "XY999999",
            "version": "XY000000.9",
        }[mutate]
    elif mutate == "wrong_voucher":
        refs[0]["voucher"] = "other"
    elif mutate == "no_voucher":
        refs[0]["voucher"] = ""
    elif mutate == "wrong_marker":
        refs[0]["coi_marker"] = False
    elif mutate == "identity":
        rows[0]["identity_status"] = "duplicate"
    elif mutate == "duplicate":
        rows[1]["values"]["COI_sequence"] = refs[0]["version"]
    a = reference_accounting(rows, refs, True)[0]
    assert a["status"] == status and a["unflagged_literal_link"] is False


def test_source_cautions_and_taxon_disagreement_do_not_rewrite_literal_evidence():
    rows, _ = records()
    refs = molecular_records(F / "references.xml")
    refs[0]["organism"] = "Different taxon"
    links = reference_accounting(rows, refs, True)
    assert links[0]["status"] == "literal_voucher_coi_match"
    assert links[0]["organism_label_equal"] is False
    assert links[1]["status"] == "literal_voucher_coi_match"
    assert links[1]["unflagged_literal_link"] is False
    assert links[1]["reference_use"] == "withheld_source_caution"


def test_ncbi_external_dtd_is_never_loaded_and_internal_entities_rejected(tmp_path):
    p = tmp_path / "refs.xml"
    raw = (F / "references.xml").read_bytes()
    dtd = b'<!DOCTYPE GBSet PUBLIC "-//NCBI//NCBI GBSeq/EN" "https://www.ncbi.nlm.nih.gov/dtd/NCBI_GBSeq.dtd">'
    p.write_bytes(b'<?xml version="1.0"?>\n' + dtd + raw)
    assert len(molecular_records(p)) == 4
    p.write_bytes(b'<!DOCTYPE GBSet [<!ENTITY x "unsafe">]>' + raw)
    with pytest.raises(ValueError):
        molecular_records(p)


def test_metadata_labels_people_and_row_order_cannot_enter_measurement_features(tmp_path):
    rows, _ = records()
    export_representation(rows, tmp_path / "a", "test")
    changed = copy.deepcopy(rows)
    for r in changed:
        for n in (
            "MOTU",
            "species",
            "family",
            "sex",
            "legators",
            "measurer",
            "locality_name",
            "COI_sequence",
            "eggs_no",
            "spines",
        ):
            r["values"][n] = "changed metadata"
    inspect_rows(changed)
    export_representation(list(reversed(changed)), tmp_path / "b", "test")
    assert (tmp_path / "a/morphology.csv").read_bytes() == (
        tmp_path / "b/morphology.csv"
    ).read_bytes()
    with (tmp_path / "a/morphology.csv").open(newline="") as f:
        reader = csv.reader(f)
        assert next(reader) == ["observation_id", *TRAITS]
    config = read_json(tmp_path / "a/dataset.json")
    assert len(config["modalities"][0]["features"]) == 25


def test_locality_ambiguities_and_partial_dates_remain_context_only():
    rows, b = records()
    for r in rows[:2]:
        r["values"].update(locality_name="NA", latitude="NA", longitude="NA")
    links, _, summary = locality_accounting(rows, b)
    assert links[0]["context_descriptor"] is links[1]["context_descriptor"] is None
    assert summary["context_descriptors_are_independence_assertions"] is False


def test_descriptive_profiles_preserve_sex_dependent_missingness():
    rows, _ = records()
    profiles = trait_profiles(rows)
    assert len(profiles) == len(TRAITS) == 25
    assert profiles["antenna_I"]["finite_values"] == 5
    assert rows[1]["values"]["antenna_I"] == "NA"


def test_headers_and_dictionary_require_exact_source_roles():
    _, b = records()
    b["sheets"]["Data"][0]["cells"]["A"]["value"] = "unrecognized"
    with pytest.raises(ValueError):
        source_rows(b)
    _, b = records()
    b["sheets"]["Descriptions"] = b["sheets"]["Descriptions"][:-1]
    with pytest.raises(ValueError):
        source_rows(b)
    assert len(COLUMNS) == 45


def test_deterministic_compressed_ledgers_and_replay(tmp_path):
    run(tmp_path, name="a")
    run(tmp_path, name="b")
    for p in (tmp_path / "a").rglob("*"):
        if p.is_file():
            assert p.read_bytes() == (tmp_path / "b" / p.relative_to(tmp_path / "a")).read_bytes()
    write_gz_rows(tmp_path / "x.gz", [dict(value="NA")])
    assert json.loads(gzip.decompress((tmp_path / "x.gz").read_bytes())) == {"value": "NA"}


def test_cli_exposes_only_audit_operation(tmp_path, capsys):
    assert (
        main(
            [
                "audit-trait-source",
                str(F / "plan.json"),
                "--source-dir",
                str(F),
                "--output-dir",
                str(tmp_path / "cli"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["empirical_fits"] == 0


@pytest.mark.parametrize(
    "raw", [b"<broken>", b"\xff", '<!DOCTYPE a [<!ENTITY x "value">]><a>&x;</a>'.encode("utf-16")]
)
def test_malformed_or_non_utf8_xml_cannot_bypass_entity_guard(raw):
    with pytest.raises(ValueError):
        xml_bytes(raw)


@pytest.mark.parametrize(
    "field,value", [("B", "Discrete numeric data"), ("C", "Length in centimetres")]
)
def test_dictionary_cannot_silently_relabel_measurement_units(field, value):
    _, book = records()
    for row in book["sheets"]["Descriptions"]:
        if row["cells"].get("A", {}).get("value") == "body_l":
            row["cells"][field]["value"] = value
    with pytest.raises(ValueError, match="millimetre"):
        source_rows(book)
