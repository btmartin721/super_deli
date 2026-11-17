"""Lazy CLI registration for a separate predictive evidence report."""

from pathlib import Path


def register(commands):
    p = commands.add_parser(
        "assess-integrative-support",
        help="Report omitted predictive support for an existing 80% run",
    )
    p.add_argument("reference_run", type=Path)
    p.add_argument("--support-config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)


def dispatch(args):
    from super_deli.io.config import read_json
    from super_deli.models.heldout_support_config import HeldoutSupportConfig
    from super_deli.workflows.heldout_support import run_heldout_support

    return run_heldout_support(
        args.reference_run,
        support_config=HeldoutSupportConfig.from_dict(read_json(args.support_config)),
        output_dir=args.output_dir,
    )
