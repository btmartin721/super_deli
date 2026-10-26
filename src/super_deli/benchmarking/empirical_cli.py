"""Fit-free empirical curation CLI, with no network or estimator imports."""

from pathlib import Path


def register(commands):
    parser = commands.add_parser(
        "curate-empirical", help="Validate and curate locked matched tables"
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    audit = commands.add_parser(
        "assess-empirical", help="Audit sampling/reference evidence without fitting"
    )
    audit.add_argument("config", type=Path)
    for flag in ("source", "protocol", "evidence", "curated-dir", "evidence-dir", "output-dir"):
        audit.add_argument("--" + flag, type=Path, required=True)
    replacement = commands.add_parser(
        "audit-replacement-empirical",
        help="Inspect locked specimen/reference sources without fitting",
    )
    replacement.add_argument("config", type=Path)
    replacement.add_argument("--source-dir", type=Path, required=True)
    replacement.add_argument("--output-dir", type=Path, required=True)
    mapping = commands.add_parser(
        "assess-correspondence",
        help="Assess documentary specimen/reference/haul correspondence without fitting",
    )
    mapping.add_argument("config", type=Path)
    for flag in ("prior-plan", "source-dir", "prior-audit-dir", "evidence-dir", "output-dir"):
        mapping.add_argument("--" + flag, type=Path, required=True)
    traits = commands.add_parser(
        "audit-trait-source", help="Audit a locked individual trait workbook without fitting"
    )
    traits.add_argument("config", type=Path)
    traits.add_argument("--source-dir", type=Path, required=True)
    traits.add_argument("--output-dir", type=Path, required=True)
    evidence = commands.add_parser(
        "assess-trait-evidence", help="Assess trait sampling/reference evidence without fitting"
    )
    evidence.add_argument("config", type=Path)
    for flag in ("prior-plan", "source-dir", "prior-audit-dir", "evidence-dir", "output-dir"):
        evidence.add_argument("--" + flag, type=Path, required=True)
    return frozenset(
        {
            "curate-empirical",
            "assess-empirical",
            "audit-replacement-empirical",
            "assess-correspondence",
            "audit-trait-source",
            "assess-trait-evidence",
        }
    )


def dispatch(args):
    from super_deli.core.empirical import EmpiricalProfile
    from super_deli.core.empirical_protocol import EmpiricalProtocol
    from super_deli.io.config import read_json
    from super_deli.workflows.empirical_curation import curate_empirical

    if args.command == "assess-trait-evidence":
        from super_deli.core.trait_audit import TraitAuditPlan
        from super_deli.core.trait_evidence import TraitEvidencePlan
        from super_deli.workflows.trait_evidence import assess_trait_evidence

        return assess_trait_evidence(
            TraitEvidencePlan.from_dict(read_json(args.config)),
            prior_plan=TraitAuditPlan.from_dict(read_json(args.prior_plan)),
            source_dir=args.source_dir,
            prior_audit_dir=args.prior_audit_dir,
            evidence_dir=args.evidence_dir,
            output_dir=args.output_dir,
        )
    if args.command == "audit-trait-source":
        from super_deli.core.trait_audit import TraitAuditPlan
        from super_deli.workflows.trait_audit import audit_trait_source

        return audit_trait_source(
            TraitAuditPlan.from_dict(read_json(args.config)),
            source_dir=args.source_dir,
            output_dir=args.output_dir,
        )
    if args.command == "assess-correspondence":
        from super_deli.core.correspondence import CorrespondencePlan
        from super_deli.core.replacement_audit import ReplacementAuditPlan
        from super_deli.workflows.correspondence import assess_correspondence

        return assess_correspondence(
            CorrespondencePlan.from_dict(read_json(args.config)),
            prior_plan=ReplacementAuditPlan.from_dict(read_json(args.prior_plan)),
            source_dir=args.source_dir,
            prior_audit_dir=args.prior_audit_dir,
            evidence_dir=args.evidence_dir,
            output_dir=args.output_dir,
        )
    if args.command == "audit-replacement-empirical":
        from super_deli.core.replacement_audit import ReplacementAuditPlan
        from super_deli.workflows.replacement_audit import audit_replacement

        return audit_replacement(
            ReplacementAuditPlan.from_dict(read_json(args.config)),
            source_dir=args.source_dir,
            output_dir=args.output_dir,
        )
    if args.command == "assess-empirical":
        from super_deli.core.empirical_evidence import EmpiricalEvidence
        from super_deli.workflows.empirical_assessment import assess_empirical

        return assess_empirical(
            args.source,
            profile=EmpiricalProfile.from_dict(read_json(args.config)),
            protocol=EmpiricalProtocol.from_dict(read_json(args.protocol)),
            evidence=EmpiricalEvidence.from_dict(read_json(args.evidence)),
            curated_dir=args.curated_dir,
            evidence_dir=args.evidence_dir,
            output_dir=args.output_dir,
        )
    return curate_empirical(
        args.source,
        profile=EmpiricalProfile.from_dict(read_json(args.config)),
        protocol=EmpiricalProtocol.from_dict(read_json(args.protocol)),
        output_dir=args.output_dir,
    )
