"""Explicit additive reference calibration/prediction; legacy commands unchanged."""

from pathlib import Path


def register(commands):
    for name in ("calibrate-reference-abstention", "predict-reference-abstention"):
        p = commands.add_parser(name, help="Opt-in reference support and withheld assignments")
        p.add_argument("config", type=Path)
        p.add_argument("--reference-run", type=Path, required=True)
        p.add_argument("--output-dir", type=Path, required=True)
        if name.startswith("calibrate"):
            p.add_argument("--training-dataset", type=Path, required=True)
            p.add_argument("--labels", type=Path, required=True)
            p.add_argument("--abstention-config", type=Path, required=True)
        else:
            p.add_argument("--calibration-run", type=Path, required=True)


def dispatch(args):
    from super_deli.io.config import load_config, read_json
    from super_deli.models.reference_abstention_config import ReferenceAbstentionConfig
    from super_deli.workflows.reference_abstention import (
        calibrate_reference_abstention,
        predict_reference_abstention,
    )

    if args.command.startswith("calibrate"):
        return calibrate_reference_abstention(
            load_config(args.training_dataset),
            load_config(args.config),
            reference_run=args.reference_run,
            calibration_labels=args.labels,
            abstention_config=ReferenceAbstentionConfig.from_dict(
                read_json(args.abstention_config)
            ),
            training_base_dir=args.training_dataset.resolve().parent,
            calibration_base_dir=args.config.resolve().parent,
            output_dir=args.output_dir,
        )
    return predict_reference_abstention(
        load_config(args.config),
        reference_run=args.reference_run,
        calibration_run=args.calibration_run,
        base_dir=args.config.resolve().parent,
        output_dir=args.output_dir,
    )
