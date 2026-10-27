"""Offline, non-executable empirical pilot protocol and cohort feasibility."""

import hashlib
from pathlib import Path

from super_deli.core.provenance import configuration_hash
from super_deli.core.trait_evidence import TraitEvidencePlan
from super_deli.core.trait_pilot import PANELS, REASONS, TraitPilotProtocol, future_point_design
from super_deli.io.config import read_json, write_json
from super_deli.io.trait_pilot import describe_roster, select_rosters
from super_deli.workflows.correspondence import verify_evidence
from super_deli.workflows.empirical_assessment import sha, verify_curated
from super_deli.workflows.trait_audit import verify_trait_sources, write_gz_rows
from super_deli.workflows.trait_evidence import _ledger, assess_trait_evidence, readiness


def assess_trait_pilot(
    plan,
    *,
    prior_plan,
    evidence_plan,
    source_dir,
    prior_audit_dir,
    evidence_dir,
    prior_assessment_dir,
    output_dir,
):
    plan = TraitPilotProtocol.from_dict(plan.to_dict())
    evidence_plan = TraitEvidencePlan.from_dict(evidence_plan.to_dict())
    out = Path(output_dir).absolute()
    for directory in (source_dir, prior_audit_dir, evidence_dir, prior_assessment_dir):
        root = Path(directory).resolve()
        if out.resolve().is_relative_to(root) or root.is_relative_to(out.resolve()):
            raise ValueError("Pilot outputs must be separate from all input roots")
    out.mkdir(parents=True, exist_ok=False)
    try:
        write_json(plan.to_dict(), out / "specification.json")
        if configuration_hash(evidence_plan.to_dict()) != plan.evidence_specification_sha256:
            raise ValueError("Evidence specification differs from bound assessment")
        assessment = Path(prior_assessment_dir)
        if (
            assessment.is_symlink()
            or sha(assessment / "checksums.json") != plan.evidence_checksums_sha256
        ):
            raise ValueError("Evidence inventory differs from bound assessment")
        original = verify_curated(assessment)
        assess_trait_evidence(
            evidence_plan,
            prior_plan=prior_plan,
            source_dir=source_dir,
            prior_audit_dir=prior_audit_dir,
            evidence_dir=evidence_dir,
            output_dir=out / "evidence-reconstruction",
        )
        rebuilt = verify_curated(out / "evidence-reconstruction")
        if original != rebuilt:
            raise ValueError("Evidence semantics differ from complete source reconstruction")
        audit = out / "evidence-reconstruction/prior-reconstruction"
        rows = _ledger(audit / "source-records.jsonl.gz")
        refs = _ledger(out / "evidence-reconstruction/specimen-evidence.jsonl.gz")
        if len({r["source_row"] for r in rows}) != len(rows) or [r["source_row"] for r in rows] != [
            r["source_row"] for r in refs
        ]:
            raise ValueError("Complete one-to-one source-row evidence accounting required")
        formulas = {
            (r["source_row"], r["field"]): r for r in read_json(audit / "formula-provenance.json")
        }
        rosters = select_rosters(rows, formulas)
        summaries = {}
        # Selection has completed for every roster before references are attached.
        for key, records in rosters.items():
            features = PANELS[key.rsplit("_", 1)[0]]
            write_gz_rows(out / (key + ".jsonl.gz"), records)
            summaries[key] = describe_roster(records, rows, refs, features)
        write_gz_rows(out / "reference-evidence.jsonl.gz", refs)
        write_json(list(formulas.values()), out / "formula-provenance.json")
        write_json(
            dict(
                panels={k: list(v) for k, v in PANELS.items()},
                exclusion_order=list(REASONS),
                family="Niphargidae",
                sexes=["male", "female"],
                missing_policy="strict_complete_positive",
                formula_policy="exclude_selected_cells_including_cached_results",
                uncertainty_policy="exclude_selected_source_numeric_issues_no_token_repair",
                allometry="not_corrected_body_length_retained",
                maturity="source_labels_not_independent_adult_verification",
                future_design=future_point_design(),
            ),
            out / "frozen-design.json",
        )
        summary = dict(
            schema_version=1,
            phase=plan.phase,
            protocol_id=plan.protocol_id,
            specification_sha256=configuration_hash(plan.to_dict()),
            status="assessed",
            source_rows=len(rows),
            rosters=summaries,
            readiness=readiness(refs),
            outcome="descriptive_rosters_assessed_inferential_pilot_no_go",
            fitting_authorized=False,
            sampling_independence_asserted=False,
            reference_species_truth_asserted=False,
            empirical_fits=0,
            bootstrap_draws=0,
            pvalues_generated=0,
            source_downloads=0,
            contact_performed=False,
            independence_requirement=(
                "Documented event/identity crosswalk and justified dependence design"
            ),
            reference_requirement=(
                "Independent specimen-linked taxonomic adjudication; PTP/ASAP remain hypotheses"
            ),
            no_ecology_manufactured=True,
            executable_analysis_exported=False,
        )
        _report(summary, out / "readiness.md")
        if original != verify_curated(assessment):
            raise ValueError("Prior assessment changed during feasibility assessment")
        if verify_curated(prior_audit_dir) != verify_curated(audit):
            raise ValueError("Prior trait audit changed during feasibility assessment")
        verify_trait_sources(prior_plan, source_dir)
        verify_evidence(evidence_plan, evidence_dir)
        write_json(summary, out / "summary.json")
        hashes = {
            p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob("*"))
            if p.is_file()
        }
        write_json(hashes, out / "checksums.json")
        return summary
    except Exception as exc:
        (out / "summary.json").unlink(missing_ok=True)
        write_json(dict(status="failed", error=str(exc)), out / "failure.json")
        raise


def _report(summary, path):
    lines = [
        "# Fit-free amphipod pilot readiness",
        "",
        "Morphology rosters are feasibility records. Independent-unit fitting, resampling,",
        "significance and species-boundary validation remain blocked. No pilot model was fitted.",
        "",
        "| Roster | Included | Documentary links | Molecular partition links |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, s in summary["rosters"].items():
        lines.append(
            f"| {name} | {s['included']} | {s['documentary_specimen_links']} | "
            f"{s['molecular_partition_links']} |"
        )
    lines += [
        "",
        "The four rosters overlap across panels and are not independent replications.",
        "Complete cases are not a random sample; "
        "source male/female labels do not verify adulthood.",
        "Body size, targeted selection and dependence may dominate morphology structure.",
        "PTP/ASAP labels remain separate hypotheses; no label was used to select specimens.",
        "",
        "Required sampling evidence: " + summary["independence_requirement"] + ".",
        "Required reference evidence: " + summary["reference_requirement"] + ".",
        "No continuous ecology, independent sampling unit or species truth was inferred.",
        "Detailed tokens, exclusions and reference ambiguities remain in local ledgers.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
