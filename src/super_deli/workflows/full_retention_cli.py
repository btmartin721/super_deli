"""Fit-free creation of an explicit matched multimodal full-retention preset."""

from pathlib import Path


def register(commands):
    p = commands.add_parser(
        "full-retention-config", help="Create a prespecified retain-all-views analysis config"
    )
    p.add_argument("config", type=Path)
    p.add_argument("--arm", required=True)
    p.add_argument("--min-cluster-size", type=int, required=True)
    p.add_argument("--output-config", type=Path, required=True)


def dispatch(args):
    from super_deli.io.config import read_json, write_json
    from super_deli.models.full_retention import full_retention_analysis
    from super_deli.models.multimodal_analysis_config import MultimodalAnalysisConfig

    original = MultimodalAnalysisConfig.from_dict(read_json(args.config))
    configured = full_retention_analysis(
        original, arm_name=args.arm, min_cluster_size=args.min_cluster_size
    )
    write_json(configured.to_dict(), args.output_config)
    selected = next(a for a in configured.arms if a.name == args.arm)
    return dict(
        status="complete",
        output_config=str(args.output_config.absolute()),
        arm=args.arm,
        fusion_policy=selected.model.fusion_policy,
        retained_dimensions=selected.model.n_pcs,
        fitting="not_performed",
        significance="not_computed",
        defaults_changed=False,
    )
