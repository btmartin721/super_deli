"""Lazy optional-dependency CLI for matched multimodal landmark-shape analysis."""

from pathlib import Path


def register(commands):
    p = commands.add_parser(
        "analyze-matched-multimodal",
        help="Compare matched fixed-landmark shape and other native PCA/GMM views",
    )
    p.add_argument("config", type=Path)
    p.add_argument("--analysis-config", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)


def dispatch(args):
    from super_deli.io.config import load_config, read_json
    from super_deli.models.multimodal_analysis_config import MultimodalAnalysisConfig
    from super_deli.workflows.multimodal_analysis import run_multimodal_analysis

    return run_multimodal_analysis(
        load_config(args.config),
        analysis_config=MultimodalAnalysisConfig.from_dict(read_json(args.analysis_config)),
        base_dir=args.config.resolve().parent,
        output_dir=args.output_dir,
    )
