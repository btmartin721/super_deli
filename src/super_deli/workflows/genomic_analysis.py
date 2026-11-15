"""Compare native genomic/phenotypic discovery on one immutable matched cohort."""

from super_deli.io.genomic_analysis import freeze_genomic_analysis
from super_deli.models.genomic_analysis_config import GenomicAnalysisConfig
from super_deli.workflows.genomic_analysis_adapters import read_genomic_arm, run_genomic_arm
from super_deli.workflows.genomic_analysis_report import write_genomic_report

LIMITATIONS = [
    "Candidate partitions do not establish species boundaries; no automatic winner or consensus.",
    "Explicit specimen matching does not establish independence or individually measured ecology.",
    "Genomic diagonal, tabular full and declared integrative covarianc"
    "e families differ; differences are not integration effects alone.",
    "Compare BIC only within each native candidate search, never across representations or arms.",
    "Membership is conditional on a selected fitted mixture, not speci"
    "es probability or resampling support.",
    "Locus-block bootstrap conditions on specimens; specimen bootstrap"
    " conditions on source loci. Do not pool their frequencies.",
    "Correlated/redundant views and arbitrary weights may emphasize sh"
    "ared signal. Conflict need not indicate species.",
    "No missing-data, grouped/unmatched, shape or neural adapter; no n"
    "ew p-values or empirical gate changes.",
]


def run_genomic_analysis(dataset, *, analysis_config, base_dir, output_dir):
    from super_deli.workflows.matched_comparison import run_comparison

    return run_comparison(
        dataset,
        analysis_config=analysis_config,
        base_dir=base_dir,
        output_dir=output_dir,
        config_type=GenomicAnalysisConfig,
        freeze_inputs=freeze_genomic_analysis,
        run_arm=run_genomic_arm,
        read_arm=read_genomic_arm,
        render=write_genomic_report,
        limitations=LIMITATIONS,
    )
