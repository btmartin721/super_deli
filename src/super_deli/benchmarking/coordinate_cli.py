"""Lazy additive CLI; frozen coordinates never become a production model adapter."""

from pathlib import Path

from super_deli.io.config import read_json


def register(commands):
    parser = commands.add_parser(
        "diagnose-neural-coordinates",
        help="Compare existing clusterers on frozen matched-view coordinates",
    )
    parser.add_argument(
        "action",
        choices=(
            "bind",
            "freeze",
            "preflight",
            "audit-preflight",
            "gate",
            "main",
            "audit-main",
            "report",
        ),
    )
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--study-config", type=Path)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--staging-bytes", type=int)


def dispatch(args):
    from super_deli.benchmarking.coordinate_config import CoordinateStudy
    from super_deli.benchmarking.coordinate_study import (
        audit_phase,
        check_frozen,
        freeze,
        gate,
        run_phase,
    )

    config = (
        CoordinateStudy.from_dict(read_json(args.study_config))
        if args.study_config
        else CoordinateStudy()
    )
    root = args.output_dir.resolve()
    source = args.archive_root.resolve()
    if args.action == "bind":
        from super_deli.benchmarking.coordinate_archive import bind

        return bind(source, root, fixture=config.fixture)
    if args.action == "freeze":
        return freeze(root, config)
    _, saved = check_frozen(root)
    if args.study_config and config != saved:
        raise ValueError("Configuration differs from freeze")
    if args.action in ("preflight", "main"):
        return run_phase(root, source, args.action, workers=args.workers)
    if args.action.startswith("audit-"):
        return audit_phase(root, source, args.action.removeprefix("audit-"))
    if args.action == "gate":
        return gate(root, staging_bytes=args.staging_bytes)
    from super_deli.benchmarking.coordinate_report import report

    return report(root, source)
