"""Lazy CLI for the resource-gated fresh shared-encoder study."""

from pathlib import Path


def register(commands):
    p = commands.add_parser(
        "benchmark-separate-gmm", help="Fresh separate-encoder shared-mixture validation"
    )
    p.add_argument(
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
    p.add_argument("--archive-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--study-config", type=Path)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--staging-bytes", type=int)


def dispatch(args):
    from super_deli.benchmarking.separate_gmm_config import SharedEncoderStudy
    from super_deli.io.config import read_json

    config = (
        SharedEncoderStudy.from_dict(read_json(args.study_config))
        if args.study_config
        else SharedEncoderStudy()
    )
    root = args.output_dir.resolve()
    if args.action == "bind":
        from super_deli.benchmarking.separate_gmm_archive import bind

        return bind(args.archive_root, root, fixture=config.fixture)
    from super_deli.benchmarking.separate_gmm_study import (
        audit_phase,
        check_frozen,
        freeze,
        gate,
        run_phase,
    )

    if args.action == "freeze":
        return freeze(root, config)
    _, stored = check_frozen(root)
    if args.study_config and config != stored:
        raise ValueError("Configuration differs from freeze")
    if args.action in ("main", "preflight"):
        return run_phase(root, args.archive_root, args.action, workers=args.workers)
    if args.action.startswith("audit-"):
        return audit_phase(root, args.archive_root, args.action.removeprefix("audit-"))
    if args.action == "gate":
        return gate(root, staging_bytes=args.staging_bytes)
    from super_deli.benchmarking.separate_gmm_report import report

    return report(root)
