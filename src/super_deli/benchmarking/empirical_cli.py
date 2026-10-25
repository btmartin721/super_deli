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
    return frozenset(
        {
            "curate-empirical",
            "assess-empirical",
            "audit-replacement-empirical",
            "assess-correspondence",
        }
    )


def dispatch(args):
    from super_deli.core.empirical import EmpiricalProfile
    from super_deli.core.empirical_protocol import EmpiricalProtocol
    from super_deli.io.config import read_json
    from super_deli.workflows.empirical_curation import curate_empirical

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
