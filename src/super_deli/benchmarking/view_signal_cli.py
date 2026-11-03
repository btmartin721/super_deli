"""Lazy opt-in archival diagnostic CLI, separate from clustering workflows."""

from pathlib import Path

from super_deli.io.config import read_json


def register(commands):
    parser = commands.add_parser(
        "diagnose-matched-view-signal",
        help="Trace signal through saved matched-view transformations",
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
    parser.add_argument("--config", type=Path)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--staging-bytes", type=int)


def dispatch(args):
    from super_deli.benchmarking.view_signal_config import ViewSignalConfig

    config = (
        ViewSignalConfig.from_dict(read_json(args.config)) if args.config else ViewSignalConfig()
    )
    root = args.output_dir.resolve()
    if args.action == "bind":
        from super_deli.benchmarking.view_signal_archive import bind

        return bind(args.archive_root, root, fixture=config.fixture)
    from super_deli.benchmarking.view_signal_study import (
        audit_phase,
        check_frozen,
        freeze,
        gate,
        run_phase,
    )

    if args.action == "freeze":
        return freeze(root, config)
    _, frozen_config = check_frozen(root)
    if args.config and config != frozen_config:
        raise ValueError("Configuration differs from freeze")
    if args.action in ("preflight", "main"):
        return run_phase(root, args.archive_root, args.action, workers=args.workers)
    if args.action.startswith("audit-"):
        return audit_phase(root, args.archive_root, args.action.removeprefix("audit-"))
    if args.action == "gate":
        return gate(root, staging_bytes=args.staging_bytes)
    from super_deli.benchmarking.view_signal_report import report

    return report(root)
