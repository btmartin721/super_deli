"""Fit-free opt-in initialization declaration; no biological labels are accepted."""

from pathlib import Path


def register(commands):
    p = commands.add_parser("truth-free-init-config", help="Declare fixed diverse GMM starts")
    p.add_argument("config", type=Path)
    p.add_argument("--arm", required=True)
    p.add_argument("--output-config", type=Path, required=True)


def dispatch(args):
    from super_deli.io.config import read_json, write_json
    from super_deli.models.initialization import truth_free_analysis
    from super_deli.models.multimodal_analysis_config import MultimodalAnalysisConfig

    source = MultimodalAnalysisConfig.from_dict(read_json(args.config))
    result = truth_free_analysis(source, arm_name=args.arm)
    write_json(result.to_dict(), args.output_config)
    return dict(
        status="complete",
        fitting="not_performed",
        significance="not_computed",
        initialization_policy="diverse_v1",
        starts=100,
        defaults_changed=False,
        output_config=str(args.output_config.absolute()),
        arm=args.arm,
    )
