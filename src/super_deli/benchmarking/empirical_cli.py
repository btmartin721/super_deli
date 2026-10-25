"""Fit-free empirical curation CLI, with no network or estimator imports."""

from pathlib import Path


def register(commands):
    parser = commands.add_parser(
        "curate-empirical", help="Validate and curate locked matched tables"
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return frozenset({"curate-empirical"})


def dispatch(args):
    from super_deli.core.empirical import EmpiricalProfile
    from super_deli.core.empirical_protocol import EmpiricalProtocol
    from super_deli.io.config import read_json
    from super_deli.workflows.empirical_curation import curate_empirical

    return curate_empirical(
        args.source,
        profile=EmpiricalProfile.from_dict(read_json(args.config)),
        protocol=EmpiricalProtocol.from_dict(read_json(args.protocol)),
        output_dir=args.output_dir,
    )
