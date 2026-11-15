"""Lazy entry point for the bounded paired initialization study."""

from pathlib import Path


def register(commands):
    p = commands.add_parser(
        "benchmark-truth-free-init", help="Validate paired initialization policies"
    )
    p.add_argument("config", type=Path)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--generator", type=Path, required=True)
    p.add_argument("--phase", choices=("preflight", "main"), required=True)


def dispatch(args):
    from super_deli.benchmarking.initialization_config import InitializationStudyConfig
    from super_deli.benchmarking.initialization_source import InitializationArchive
    from super_deli.benchmarking.initialization_study import study
    from super_deli.io.config import read_json

    config = InitializationStudyConfig.from_dict(read_json(args.config))
    return study(
        InitializationArchive(args.source, config),
        config,
        args.output_dir,
        args.phase,
        args.generator,
    )
