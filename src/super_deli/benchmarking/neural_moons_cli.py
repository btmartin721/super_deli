"""Lazy diagnostic CLI; ordinary imports do not require numerical libraries."""

from pathlib import Path

from super_deli.io.config import read_json


def register(commands):
    parser = commands.add_parser(
        "diagnose-neural-moons", help="Archival neural-representation and density diagnosis"
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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--staging-bytes", type=int)


def dispatch(args):
    from super_deli.benchmarking.neural_moons_config import NeuralMoonsConfig

    config = (
        NeuralMoonsConfig.from_dict(read_json(args.config)) if args.config else NeuralMoonsConfig()
    )
    root = args.output_dir.resolve()
    if args.action == "bind":
        from super_deli.benchmarking.neural_moons_archive import audit_archive

        return audit_archive(args.archive_root, output_dir=root, fixture=config.fixture)
    from super_deli.benchmarking.neural_moons_study import (
        audit_phase,
        check_frozen,
        freeze,
        gate,
        run_phase,
    )

    if args.action == "freeze":
        return freeze(root, config)
    _, frozen_config = check_frozen(root)
    if args.config and frozen_config != config:
        raise ValueError("Configuration differs from its freeze")
    if args.action in {"preflight", "main"}:
        return run_phase(root, args.archive_root, args.action, workers=args.workers)
    if args.action.startswith("audit-"):
        return audit_phase(root, args.archive_root, args.action.removeprefix("audit-"))
    if args.action == "gate":
        if args.staging_bytes is None:
            raise ValueError("Measure and supply --staging-bytes")
        return gate(root, staging_bytes=args.staging_bytes)
    from super_deli.benchmarking.neural_moons_report import report

    return report(root)
