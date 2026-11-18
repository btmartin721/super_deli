"""Command-line entry points over the public fit-free analysis API."""

from pathlib import Path


def register(commands):
    analysis = commands.add_parser("analyze", help="Plan an analysis or read a sealed archive")
    operations = analysis.add_subparsers(dest="operation", required=True)
    plan = operations.add_parser("plan", help="Resolve a declared analysis design")
    plan.add_argument("dataset", type=Path)
    plan.add_argument("--modality", action="append", required=True)
    plan.add_argument(
        "--target-level", choices=("specimen", "population", "locality"), required=True
    )
    plan.add_argument("--missing-views", choices=("error", "complete_cases"), default="error")
    plan.add_argument("--aggregate", action="append", default=[], metavar="MODALITY=POLICY")
    report = operations.add_parser("report", help="Render a sealed matched-multimodal archive")
    report.add_argument("run_dir", type=Path)
    report.add_argument("--study-manifest", type=Path, required=True)
    report.add_argument("--output-dir", type=Path, required=True)
    report.add_argument("--boundary-arm")
    report.add_argument("--boundary-groups", nargs=2, metavar=("LEFT", "RIGHT"))
    report.add_argument("--specimen-pair", nargs=2, metavar=("FIRST", "SECOND"))

    compare = operations.add_parser(
        "compare-hypotheses", help="Compare declared specimen partitions"
    )
    compare.add_argument("run_dir", type=Path)
    compare.add_argument("--study-manifest", type=Path, required=True)
    compare.add_argument("--hypotheses", type=Path, required=True)
    compare.add_argument("--output-dir", type=Path, required=True)


def dispatch(args):
    from . import compare_hypotheses_from_archive, plan_analysis, report_from_archive

    if args.operation == "plan":
        aggregation = {}
        for entry in args.aggregate:
            name, equals, policy = entry.partition("=")
            if not equals or name in aggregation:
                raise ValueError("Use one --aggregate MODALITY=POLICY per selected modality")
            aggregation[name] = policy
        return plan_analysis(
            args.dataset,
            modalities=args.modality,
            target_level=args.target_level,
            missing_views=args.missing_views,
            aggregation=aggregation,
        ).to_dict()
    if args.operation == "compare-hypotheses":
        result = compare_hypotheses_from_archive(
            args.run_dir,
            study_manifest=args.study_manifest,
            hypotheses=args.hypotheses,
            output_dir=args.output_dir,
        )
    else:
        result = report_from_archive(
            args.run_dir,
            study_manifest=args.study_manifest,
            output_dir=args.output_dir,
            boundary_arm=args.boundary_arm,
            boundary_groups=tuple(args.boundary_groups) if args.boundary_groups else None,
            specimen_pair=tuple(args.specimen_pair) if args.specimen_pair else None,
        )
    return {"json_path": str(result.json_path), "html_path": str(result.html_path)}
