"""Explicit class-local support commands; predecessor commands are untouched."""

from pathlib import Path


def register(commands):
    for name in ("calibrate-local-reference-support", "predict-local-reference-support"):
        p = commands.add_parser(name, help="Opt-in class-local reference support")
        p.add_argument("config", type=Path)
        p.add_argument("--reference-run", type=Path, required=True)
        p.add_argument("--output-dir", type=Path, required=True)
        if name.startswith("calibrate"):
            p.add_argument("--training-dataset", type=Path, required=True)
            p.add_argument("--labels", type=Path, required=True)
            p.add_argument("--support-config", type=Path, required=True)
        else:
            p.add_argument("--calibration-run", type=Path, required=True)


def dispatch(args):
    from super_deli.io.config import load_config, read_json
    from super_deli.models.local_reference_config import LocalReferenceSupportConfig
    from super_deli.workflows.local_reference_support import (
        calibrate_local_reference_support,
        predict_local_reference_support,
    )

    if args.command.startswith("calibrate"):
        return calibrate_local_reference_support(
            load_config(args.training_dataset),
            load_config(args.config),
            reference_run=args.reference_run,
            calibration_labels=args.labels,
            support_config=LocalReferenceSupportConfig.from_dict(read_json(args.support_config)),
            training_base_dir=args.training_dataset.resolve().parent,
            calibration_base_dir=args.config.resolve().parent,
            output_dir=args.output_dir,
        )
    return predict_local_reference_support(
        load_config(args.config),
        reference_run=args.reference_run,
        calibration_run=args.calibration_run,
        base_dir=args.config.resolve().parent,
        output_dir=args.output_dir,
    )
