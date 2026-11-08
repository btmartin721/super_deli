"""Lazy opt-in calibration entry point; existing neural workflows are unchanged."""

from pathlib import Path


def register(commands):
    p = commands.add_parser(
        "calibrate-matched-separate-gmm", help="Full-pipeline fitted raw-Gaussian neural test"
    )
    p.add_argument("config", type=Path)
    p.add_argument("--calibration-config", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)


def dispatch(args):
    from super_deli.io.config import load_config, read_json
    from super_deli.models.neural_calibration_config import NeuralCalibrationConfig
    from super_deli.workflows.neural_calibration import run_neural_calibration

    return run_neural_calibration(
        load_config(args.config),
        config=NeuralCalibrationConfig.from_dict(read_json(args.calibration_config)),
        base_dir=args.config.resolve().parent,
        output_dir=args.output_dir,
    )
