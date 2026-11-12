"""Source-bound lifecycle for the bounded nonlinear/reference-classifier benchmark."""

from pathlib import Path


def register(commands):
    p = commands.add_parser(
        "benchmark-residual-nonlinear", help="Bounded supervised/discovery simulation comparison"
    )
    p.add_argument("action", choices=("bind", "freeze", "run", "audit", "gate", "report"))
    p.add_argument("--archive-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--study-config", type=Path)
    p.add_argument("--phase", choices=("preflight", "main"), default="preflight")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--staging-bytes", type=int, default=0)
    p.add_argument("--fixture", action="store_true")


def dispatch(args):
    from super_deli.benchmarking.residual_config import ResidualStudy
    from super_deli.benchmarking.residual_report import report
    from super_deli.benchmarking.residual_source import bind
    from super_deli.benchmarking.residual_study import engine, gate
    from super_deli.io.config import read_json

    root, source = args.output_dir, args.archive_root
    if args.action == "bind":
        return bind(source, root, fixture=args.fixture)
    study = engine(root, source)
    if args.action == "freeze":
        return study.freeze(
            ResidualStudy.from_dict(read_json(args.study_config))
            if args.study_config
            else ResidualStudy(fixture=args.fixture)
        )
    if args.action == "run":
        return study.run(args.phase, workers=args.workers)
    if args.action == "audit":
        return study.audit(args.phase)
    if args.action == "gate":
        return gate(root, source, staging_bytes=args.staging_bytes)
    return report(root, source)
