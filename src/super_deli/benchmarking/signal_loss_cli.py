"""Lazy source-bound point-diagnostic entry point, separate from production analysis."""

from pathlib import Path


def register(commands):
    p = commands.add_parser(
        "diagnose-multimodal-signal", help="Bounded diagnostic of archived multimodal signal loss"
    )
    p.add_argument("config", type=Path)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--phase", choices=("audit", "preflight", "main"), default="audit")
    p.add_argument("--generator", type=Path)


def dispatch(args):
    from super_deli.benchmarking.signal_loss_config import SignalLossConfig
    from super_deli.benchmarking.signal_loss_source import SealedStudy
    from super_deli.benchmarking.signal_loss_study import audit_source, study
    from super_deli.io.config import read_json

    config = SignalLossConfig.from_dict(read_json(args.config))
    source = SealedStudy(args.source, config)
    if args.phase == "audit":
        return audit_source(source, args.output_dir)
    if args.generator is None:
        raise ValueError("Scientific phases require the frozen invented generator path")
    return study(
        source, config, output_dir=args.output_dir, phase=args.phase, generator_path=args.generator
    )
