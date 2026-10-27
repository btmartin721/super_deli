"""Fit-free raw cohort selection, source binding and unsupported evidence gates."""

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from super_deli.cli import main
from super_deli.core.provenance import configuration_hash
from super_deli.core.trait_audit import COLUMNS, TRAITS, TraitAuditPlan
from super_deli.core.trait_evidence import TraitEvidencePlan
from super_deli.core.trait_pilot import CORE19, PANELS, TraitPilotProtocol, future_point_design
from super_deli.io.config import read_json, write_json
from super_deli.io.trait_pilot import describe_roster, select_rosters
from super_deli.workflows.empirical_assessment import verify_curated
from super_deli.workflows.trait_audit import audit_trait_source
from super_deli.workflows.trait_evidence import assess_trait_evidence
from super_deli.workflows.trait_pilot import assess_trait_pilot

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "examples/amphipod-audit/fixture"
EVIDENCE = ROOT / "examples/amphipod-gates/fixture"


def row(index=2):
    values = dict.fromkeys(COLUMNS, "")
    values.update({f: "1.25" for f in TRAITS})
    values.update(voucher=f"invented_{index}", family="Niphargidae", sex="male")
    return dict(
        source_row=index,
        unit_id=f"unit_{index}",
        identity_status="unique",
        values=values,
        numeric_values={f: 1.25 for f in TRAITS},
        numeric_issues={},
    )


def ref(index=2):
    return dict(
        source_row=index,
        molecular_partition_link=True,
        raw_molecular_partitions={"ptp": "p1", "asap": "a7"},
        historical_motu="old1",
        correspondence_status="documented",
        documented_specimen_link=True,
        source_cautions=[],
        genbank_conflict=False,
        contradictory_voucher_records=[],
    )


def test_panels_are_fixed_ordered_raw_somatic_measurements():
    assert len(CORE19) == 19 and len(PANELS["somatic24"]) == 24
    assert PANELS["somatic24"] == (*CORE19, *TRAITS[19:24])
    assert "egg_diam_avg" not in PANELS["somatic24"]


@pytest.mark.parametrize("sex", ["male subadult", "female subadult", "juvenile", "NA", "", "Male"])
def test_no_stage_or_sex_inference(sex):
    r = row()
    r["values"]["sex"] = sex
    assert all(not entries[0]["included"] for entries in select_rosters([r], {}).values())


@pytest.mark.parametrize("status", ["missing", "duplicate", "invalid"])
def test_identity_must_be_unique(status):
    r = row()
    r["identity_status"] = status
    assert select_rosters([r], {})["core19_male"][0]["primary_exclusion"] == "ambiguous_identity"


def test_fixed_panel_selection_formula_positive_and_raw_value_preservation():
    r = row()
    r["numeric_values"]["upI_basis"] = None
    r["values"]["upI_basis"] = "NA"
    selected = select_rosters([r], {})
    assert selected["core19_male"][0]["included"]
    assert not selected["somatic24_male"][0]["included"]
    assert selected["somatic24_male"][0]["raw_selected_values"]["upI_basis"] == "NA"
    assert not select_rosters([r], {(2, "body_l"): {}})["core19_male"][0]["included"]
    r["numeric_values"]["body_l"] = 0
    assert "nonpositive_measurement" in select_rosters([r], {})["core19_male"][0]["reasons"]


@pytest.mark.parametrize(
    "token,issue",
    [("0.2?", "uncertain"), ("NANA", "invalid"), ("1,2,3", "invalid"), ("nan", "nonfinite")],
)
def test_no_uncertain_token_repair(token, issue):
    r = row()
    r["values"]["body_l"] = token
    r["numeric_values"]["body_l"] = None
    r["numeric_issues"]["body_l"] = issue
    entry = select_rosters([r], {})["core19_male"][0]
    assert entry["primary_exclusion"] == "uncertain_measurement"
    assert entry["raw_selected_values"]["body_l"] == token


def test_reference_and_geography_cannot_change_membership():
    r = row()
    baseline = select_rosters([r], {})
    for name in ("MOTU", "species", "COI_sequence", "locality_name", "survey_date", "remarks"):
        r["values"][name] = "contradictory_reference_or_context"
    r["reference_cautions"] = ["possibly swapped"]
    assert select_rosters([r], {}) == baseline


def test_all_rows_accounted_with_overlapping_reasons_and_fixed_first_exclusion():
    a, b = row(2), row(3)
    b["identity_status"] = "duplicate"
    b["values"]["family"] = "Typhlogammaridae"
    entries = select_rosters([a, b], {})["core19_male"]
    s = describe_roster(entries, [a, b], [ref(2), ref(3)], CORE19)
    assert s["included"] == s["excluded"] == 1
    assert s["first_exclusion_counts"] == {"ambiguous_identity": 1}
    assert s["overlapping_reason_counts"]["outside_family"] == 1
    assert s["molecular_partition_links"] == 1


def test_constant_duplicate_and_separate_reference_namespaces():
    rows = [row(2), row(3)]
    refs = [ref(2), ref(3)]
    refs[1]["raw_molecular_partitions"]["ptp"] = ["p1", "p2"]
    s = describe_roster(select_rosters(rows, {})["core19_male"], rows, refs, CORE19)
    assert s["constant_traits"] == list(CORE19)
    assert s["duplicate_numeric_vector_groups"] == 1
    assert s["duplicate_numeric_vector_rows"] == 2
    assert s["raw_partition_label_counts"] == {"asap": {"a7": 2}, "ptp": {"p1": 2, "p2": 1}}
    assert s["nominal_pca_rank_upper_bound"] == 1


def test_empty_roster_is_no_go_without_fitting():
    s = describe_roster([], [], [], CORE19)
    assert s["included"] == s["nominal_pca_rank_upper_bound"] == 0
    assert s["source_only_morphology_roster"] == "empty_under_fixed_policy"
    assert s["constant_traits"] == []


def protocol():
    return TraitPilotProtocol("invented_pilot", "a" * 64, "b" * 64)


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("phase", "fit"),
        ("protocol_id", "../person"),
        ("evidence_checksums_sha256", "bad"),
        ("fitting_authorized", True),
        ("independent_units_confirmed", True),
        ("reference_species_truth_confirmed", True),
        ("contact_policy", "email"),
        ("cohort_policy", "maximize_coverage"),
        ("reference_policy", "species_truth"),
    ],
)
def test_protocol_rejects_promotion_or_undeclared_policy(key, value):
    with pytest.raises(ValueError):
        TraitPilotProtocol.from_dict(dict(protocol().to_dict(), **{key: value}))


def test_protocol_is_immutable_and_future_design_is_non_executable():
    p = protocol()
    assert TraitPilotProtocol.from_dict(p.to_dict()) == p
    with pytest.raises(FrozenInstanceError):
        p.phase = "fit"
    with pytest.raises(ValueError):
        TraitPilotProtocol.from_dict(dict(p.to_dict(), extra=True))
    d = future_point_design()
    assert not d["fitting_authorized"] and not d["independent_units_confirmed"]
    assert d["bootstrap_draws"] == d["significance_draws"] == d["subsampling_draws"] == 0
    assert "assume_independent_units" not in json.dumps(d)


def world(tmp_path):
    prior = TraitAuditPlan.from_dict(read_json(PRIOR / "plan.json"))
    ep = TraitEvidencePlan.from_dict(read_json(EVIDENCE / "plan.json"))
    audit = tmp_path / "audit"
    assessment = tmp_path / "assessment"
    audit_trait_source(prior, source_dir=PRIOR, output_dir=audit)
    assess_trait_evidence(
        ep,
        prior_plan=prior,
        source_dir=PRIOR,
        prior_audit_dir=audit,
        evidence_dir=EVIDENCE,
        output_dir=assessment,
    )
    p = TraitPilotProtocol(
        "invented_pilot",
        configuration_hash(ep.to_dict()),
        hashlib.sha256((assessment / "checksums.json").read_bytes()).hexdigest(),
    )
    kwargs = dict(
        prior_plan=prior,
        evidence_plan=ep,
        source_dir=PRIOR,
        prior_audit_dir=audit,
        evidence_dir=EVIDENCE,
        prior_assessment_dir=assessment,
    )
    return p, kwargs


def test_full_source_bound_workflow_replay_and_closed_gates(tmp_path):
    p, kwargs = world(tmp_path)
    result = assess_trait_pilot(p, **kwargs, output_dir=tmp_path / "pilot")
    again = assess_trait_pilot(p, **kwargs, output_dir=tmp_path / "replay")
    from super_deli.benchmarking.trait_pilot_audit import audit_trait_pilot

    assert audit_trait_pilot(tmp_path / "pilot")["roster_row_decisions"] == 24
    assert result == again
    assert verify_curated(tmp_path / "pilot") == verify_curated(tmp_path / "replay")
    assert result["source_rows"] == 6
    assert (
        result["empirical_fits"] == result["pvalues_generated"] == result["source_downloads"] == 0
    )
    assert result["contact_performed"] is result["fitting_authorized"] is False
    for name in (
        "independent_unit_fitting",
        "grouped_predictive_screen",
        "resampling",
        "significance",
        "species_boundary_validation",
    ):
        assert result["readiness"][name]["status"] == "blocked"
    for entries in result["rosters"].values():
        assert entries["included"] + entries["excluded"] == 6
    assert not list((tmp_path / "pilot").glob("dataset.json"))


@pytest.mark.parametrize("mode", ["checksum", "specification", "extra", "rehashed_semantics"])
def test_assessment_tampering_fails_closed_with_failure_evidence(tmp_path, mode):
    p, kwargs = world(tmp_path)
    root = kwargs["prior_assessment_dir"]
    if mode == "specification":
        p = replace(p, evidence_specification_sha256="c" * 64)
    elif mode == "checksum":
        p = replace(p, evidence_checksums_sha256="c" * 64)
    elif mode == "extra":
        (root / "unlisted.json").write_text("{}")
    else:
        target = root / "summary.json"
        v = read_json(target)
        v["source_rows"] = 999
        target.write_text(json.dumps(v, indent=2, sort_keys=True) + "\n")
        checks = read_json(root / "checksums.json")
        checks["summary.json"] = hashlib.sha256(target.read_bytes()).hexdigest()
        (root / "checksums.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
        p = replace(
            p,
            evidence_checksums_sha256=hashlib.sha256(
                (root / "checksums.json").read_bytes()
            ).hexdigest(),
        )
    with pytest.raises(ValueError):
        assess_trait_pilot(p, **kwargs, output_dir=tmp_path / "failed")
    assert (tmp_path / "failed/failure.json").exists()
    assert not (tmp_path / "failed/summary.json").exists()


def test_cli_dispatch(tmp_path, capsys):
    p, kwargs = world(tmp_path)
    write_json(p.to_dict(), tmp_path / "plan.json")
    args = [
        "assess-trait-pilot",
        str(tmp_path / "plan.json"),
        "--prior-plan",
        str(PRIOR / "plan.json"),
        "--evidence-plan",
        str(EVIDENCE / "plan.json"),
        "--output-dir",
        str(tmp_path / "cli"),
    ]
    for name in ("source_dir", "prior_audit_dir", "evidence_dir", "prior_assessment_dir"):
        args.extend(["--" + name.replace("_", "-"), str(kwargs[name])])
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["empirical_fits"] == 0


def test_unselected_formula_or_missing_egg_does_not_remove_somatic_specimen():
    r = row()
    r["numeric_values"]["egg_diam_avg"] = None
    r["values"]["egg_diam_avg"] = "NA"
    entries = select_rosters([r], {(2, "egg_diam_avg"): {}})
    assert entries["core19_male"][0]["included"]
    assert entries["somatic24_male"][0]["included"]


def test_input_output_overlap_rejected_without_mutating_inputs(tmp_path):
    p, kwargs = world(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        assess_trait_pilot(p, **kwargs, output_dir=kwargs["prior_audit_dir"] / "nested")
    assert not (kwargs["prior_audit_dir"] / "nested").exists()


def test_late_prior_change_withholds_completed_summary(tmp_path, monkeypatch):
    import super_deli.workflows.trait_pilot as workflow

    p, kwargs = world(tmp_path)
    original = workflow.describe_roster

    def change(*args):
        (kwargs["prior_audit_dir"] / "unexpected.json").write_text("{}")
        return original(*args)

    monkeypatch.setattr(workflow, "describe_roster", change)
    with pytest.raises(ValueError, match="Unlisted"):
        assess_trait_pilot(p, **kwargs, output_dir=tmp_path / "late-failure")
    assert (tmp_path / "late-failure/failure.json").exists()
    assert not (tmp_path / "late-failure/summary.json").exists()
