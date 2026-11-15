"""Lazy CLI registration for matched genomic–phenotypic discovery."""

from pathlib import Path


def register(commands):
    p = commands.add_parser(
        "analyze-matched-genomic", help="Compare native genomic/phenotypic PCA/GMM arms"
    )
    p.add_argument("config", type=Path)
    p.add_argument("--analysis-config", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)


def dispatch(args):
    from super_deli.io.config import load_config, read_json
    from super_deli.models.genomic_analysis_config import GenomicAnalysisConfig
    from super_deli.workflows.genomic_analysis import run_genomic_analysis

    return run_genomic_analysis(
        load_config(args.config),
        analysis_config=GenomicAnalysisConfig.from_dict(read_json(args.analysis_config)),
        base_dir=args.config.resolve().parent,
        output_dir=args.output_dir,
    )
