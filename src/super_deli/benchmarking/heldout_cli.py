"""Lazy entry point for the bounded, sealed-source validation protocol."""

from pathlib import Path


def register(commands):
    p = commands.add_parser(
        "benchmark-heldout-support", help="Validate omitted prediction and rare-group feasibility"
    )
    p.add_argument("config", type=Path)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--generator", type=Path, required=True)
    p.add_argument("--phase", choices=("preflight", "main"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)


def dispatch(args):
    from super_deli.benchmarking.heldout_config import HeldoutStudyConfig
    from super_deli.benchmarking.heldout_source import HeldoutArchive
    from super_deli.benchmarking.heldout_study import study
    from super_deli.io.config import read_json

    config = HeldoutStudyConfig.from_dict(read_json(args.config))
    return study(
        HeldoutArchive(args.source, config), config, args.output_dir, args.phase, args.generator
    )
