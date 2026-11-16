"""Lazy entry point for opt-in whole-individual subsampling of a native point run."""

from pathlib import Path


def register(commands):
    p = commands.add_parser(
        "subsample-integrative",
        help="Attach 80% whole-individual sensitivity to a native full-retention run",
    )
    p.add_argument("reference_run", type=Path)
    p.add_argument("--stability-config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)


def dispatch(args):
    from super_deli.io.config import read_json
    from super_deli.models.individual_subsample_config import IndividualSubsampleConfig
    from super_deli.workflows.individual_subsampling import run_individual_subsampling

    return run_individual_subsampling(
        args.reference_run,
        stability_config=IndividualSubsampleConfig.from_dict(read_json(args.stability_config)),
        output_dir=args.output_dir,
    )
