"""Lazy CLI for the separately bounded scientific validation."""

from pathlib import Path


def register(commands):
    p = commands.add_parser(
        "benchmark-individual-subsampling",
        help="Run frozen paired bootstrap/subsampling validation",
    )
    p.add_argument("config", type=Path)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--generator", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--phase", choices=("preflight", "main"), required=True)


def dispatch(args):
    from super_deli.benchmarking.individual_subsample_config import IndividualSubsampleStudyConfig
    from super_deli.benchmarking.individual_subsample_source import IndividualSubsampleArchive
    from super_deli.benchmarking.individual_subsample_study import study
    from super_deli.io.config import read_json

    c = IndividualSubsampleStudyConfig.from_dict(read_json(args.config))
    source = IndividualSubsampleArchive(args.source, c)
    return study(source, c, args.output_dir, args.phase, args.generator)
