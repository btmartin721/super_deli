"""Lazy benchmark-only entry point; no production resampling option."""

from pathlib import Path


def register(commands):
    p = commands.add_parser(
        "diagnose-bootstrap-splitting", help="Run the bounded resampling/eligibility diagnosis"
    )
    p.add_argument("config", type=Path)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--generator", type=Path, required=True)
    p.add_argument("--phase", choices=("cache", "preflight", "main"), required=True)


def dispatch(args):
    from super_deli.benchmarking.bootstrap_split_config import BootstrapSplitConfig
    from super_deli.benchmarking.bootstrap_split_source import BootstrapSplitArchive, cached_trace
    from super_deli.benchmarking.bootstrap_split_study import study
    from super_deli.io.config import read_json

    config = BootstrapSplitConfig.from_dict(read_json(args.config))
    source = BootstrapSplitArchive(args.source, config)
    if args.phase == "cache":
        return cached_trace(source, args.output_dir / "cache")
    return study(source, config, args.output_dir, args.phase, args.generator)
