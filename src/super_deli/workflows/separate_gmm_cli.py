"""Lazy standalone entry point for the opt-in shared-GMM neural adapter."""

from pathlib import Path


def register(commands):
    parser = commands.add_parser(
        "cluster-matched-separate-gmm", help="Separate matched encoders and shared Gaussian mixture"
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)


def dispatch(args):
    from super_deli.io.config import load_config, read_json
    from super_deli.models.separate_gmm_config import SeparateGMMConfig
    from super_deli.workflows.separate_gmm import run_separate_gmm

    return run_separate_gmm(
        load_config(args.config),
        model_config=SeparateGMMConfig.from_dict(read_json(args.model_config)),
        base_dir=args.config.resolve().parent,
        output_dir=args.output_dir,
    )
