"""Lazy opt-in supervised training and prediction commands."""

from pathlib import Path


def register(commands):
    for name in ("train-matched-residual", "predict-matched-residual"):
        parser = commands.add_parser(
            name, help="Supervised matched-view reference assignment; novelty unassessed"
        )
        parser.add_argument("config", type=Path)
        parser.add_argument("--output-dir", required=True, type=Path)
        if name.startswith("train"):
            parser.add_argument("--labels", required=True, type=Path)
            parser.add_argument("--model-config", required=True, type=Path)
        else:
            parser.add_argument("--reference-run", required=True, type=Path)


def dispatch(args):
    from super_deli.io.config import load_config, read_json
    from super_deli.models.residual_config import ResidualConfig
    from super_deli.workflows.residual_reference import (
        predict_matched_residual,
        train_matched_residual,
    )

    common = dict(base_dir=args.config.resolve().parent, output_dir=args.output_dir)
    if args.command == "train-matched-residual":
        return train_matched_residual(
            load_config(args.config),
            labels_path=args.labels,
            model_config=ResidualConfig.from_dict(read_json(args.model_config)),
            **common,
        )
    return predict_matched_residual(
        load_config(args.config), reference_run=args.reference_run, **common
    )
