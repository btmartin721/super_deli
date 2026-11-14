"""Lazy optional-dependency CLI for one matched discovery comparison."""

from pathlib import Path


def register(commands):
    p = commands.add_parser(
        "analyze-matched-tabular",
        help="Compare prespecified matched morphology/ecology discovery arms",
    )
    p.add_argument("config", type=Path)
    p.add_argument("--analysis-config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)


def dispatch(args):
    from super_deli.io.config import load_config, read_json
    from super_deli.models.matched_analysis_config import MatchedAnalysisConfig
    from super_deli.workflows.matched_analysis import run_matched_analysis

    return run_matched_analysis(
        load_config(args.config),
        analysis_config=MatchedAnalysisConfig.from_dict(read_json(args.analysis_config)),
        base_dir=args.config.resolve().parent,
        output_dir=args.output_dir,
    )
