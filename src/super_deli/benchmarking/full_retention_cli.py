"""Lazy entry point for the frozen full-retention validation protocol."""

from pathlib import Path


def register(commands):
    p = commands.add_parser(
        "benchmark-full-retention", help="Run the bounded full-retention validation"
    )
    p.add_argument("config", type=Path)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--generator", type=Path, required=True)
    p.add_argument("--phase", choices=("preflight", "main"), required=True)


def dispatch(args):
    from super_deli.benchmarking.full_retention_config import FullRetentionStudyConfig
    from super_deli.benchmarking.full_retention_source import RetentionArchive
    from super_deli.benchmarking.full_retention_study import study
    from super_deli.io.config import read_json

    config = FullRetentionStudyConfig.from_dict(read_json(args.config))
    source = RetentionArchive(args.source, config)
    return study(source, config, args.output_dir, args.phase, args.generator)
