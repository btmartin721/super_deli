"""Lazy additive entry points for paired neural clustering and its frozen study."""

from pathlib import Path

from super_deli.io.config import load_config, read_json


def register(commands):
    p = commands.add_parser(
        "cluster-matched-neural", help="View-balanced matched individual morphology/ecology"
    )
    p.add_argument("config", type=Path)
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p = commands.add_parser("benchmark-matched-neural", help="Gated matched neural study")
    p.add_argument(
        "action",
        choices=("freeze", "preflight", "audit-preflight", "gate", "main", "audit-main", "report"),
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--study-config", type=Path)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--staging-bytes", type=int)


def dispatch(args):
    if args.command == "cluster-matched-neural":
        from super_deli.models.matched_neural_config import MatchedNeuralConfig
        from super_deli.workflows.matched_neural import run_matched_neural

        return run_matched_neural(
            load_config(args.config),
            model_config=MatchedNeuralConfig.from_dict(read_json(args.model_config)),
            base_dir=args.config.resolve().parent,
            output_dir=args.output_dir,
        )
    from super_deli.benchmarking.matched_study import dispatch_study

    return dispatch_study(args)
