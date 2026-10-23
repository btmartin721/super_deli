"""Thin command-line interface over the data validation API."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

from super_deli import __version__
from super_deli.core.provenance import collect_provenance
from super_deli.core.schema import ConfigError
from super_deli.core.validation import validate_dataset
from super_deli.io.config import load_config, read_json, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate, ingest, and analyze Super Deli datasets"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Check declarations and local file references")
    validate.add_argument("config", type=Path, help="Versioned dataset JSON configuration")
    validate.add_argument("--report", type=Path, help="Also save JSON to a new file")
    validate.add_argument(
        "--hash-inputs", action="store_true", help="SHA-256 every directly referenced input file"
    )
    ingest = commands.add_parser("ingest-genomic", help="Load a genomic source through SNPio")
    ingest.add_argument("config", type=Path, help="Versioned dataset JSON configuration")
    ingest.add_argument(
        "--modality", help="Genomic modality name; required when more than one exists"
    )
    ingest.add_argument(
        "--output-dir", type=Path, required=True, help="New directory for staged data and outputs"
    )
    tabular = commands.add_parser(
        "ingest-tabular", help="Load typed morphology/ecology CSV/TSV tables"
    )
    tabular.add_argument("config", type=Path, help="Versioned dataset JSON configuration")
    tabular.add_argument(
        "--modality",
        action="append",
        help="Select a table by name; repeat to select multiple (default: all tables)",
    )
    tabular.add_argument(
        "--output-dir", type=Path, required=True, help="New directory for staged tables and outputs"
    )
    morphometric = commands.add_parser(
        "ingest-morphometric", help="Load fixed landmarks from TPS/coordinate CSV"
    )
    morphometric.add_argument("config", type=Path, help="Versioned dataset JSON configuration")
    morphometric.add_argument(
        "--modality", action="append", help="Select TPS/landmark CSV by name; repeat for multiple"
    )
    morphometric.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for staged sources and landmarks",
    )
    images = commands.add_parser(
        "ingest-images", help="Decode PDF, PNG, JPEG and HEIC/HEIF image manifests"
    )
    images.add_argument("config", type=Path, help="Versioned dataset JSON configuration")
    images.add_argument(
        "--modality", action="append", help="Select an image manifest by name; repeat for multiple"
    )
    images.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for staged sources and PNG derivatives",
    )
    detection = commands.add_parser(
        "detect-landmarks", help="Locate explicitly named visible markers"
    )
    detection.add_argument("config", type=Path, help="Dataset JSON with an image_manifest modality")
    detection.add_argument(
        "--detector-config", type=Path, required=True, help="Named marker rules JSON"
    )
    detection.add_argument(
        "--modality", help="Select one image manifest; required if several exist"
    )
    detection.add_argument(
        "--output-dir", type=Path, required=True, help="New detection output directory"
    )
    shape = commands.add_parser(
        "shape-pca", help="Align fixed landmarks with GPA and fit/project shape PCA"
    )
    shape.add_argument("config", type=Path, help="Dataset JSON with TPS/coordinate CSV")
    shape.add_argument(
        "--model-config", type=Path, required=True, help="Shape analysis policies JSON"
    )
    shape.add_argument("--modality", help="Select exactly one coordinate modality")
    shape.add_argument(
        "--reference-run", type=Path, help="Project using a saved training run without refitting"
    )
    shape.add_argument(
        "--output-dir", type=Path, required=True, help="New alignment/PCA output directory"
    )
    clustering = commands.add_parser("cluster-shape", help="Cluster shape PCs with unit bootstrap")
    clustering.add_argument("config", type=Path, help="Dataset JSON with TPS/coordinate CSV")
    clustering.add_argument("--model-config", type=Path, required=True)
    clustering.add_argument("--modality", help="Select exactly one coordinate modality")
    clustering.add_argument("--output-dir", type=Path, required=True)
    tabular_cluster = commands.add_parser(
        "cluster-tabular", help="Cluster numeric traits/environment"
    )
    tabular_cluster.add_argument("config", type=Path)
    tabular_cluster.add_argument("--model-config", type=Path, required=True)
    tabular_cluster.add_argument("--modality", help="Select exactly one morphology/ecology table")
    tabular_cluster.add_argument("--output-dir", type=Path, required=True)
    integrate = commands.add_parser(
        "cluster-integrative", help="Integrate linked modality PC profiles"
    )
    integrate.add_argument("config", type=Path)
    integrate.add_argument("--model-config", type=Path, required=True)
    integrate.add_argument("--output-dir", type=Path, required=True)
    benchmark = commands.add_parser(
        "benchmark-integrative", help="Run a declared truth-separated integrative simulation study"
    )
    benchmark.add_argument("config", type=Path, help="Versioned benchmark JSON configuration")
    benchmark.add_argument("--output-dir", type=Path, required=True)
    study_analysis = commands.add_parser(
        "analyze-integrative-study", help="Audit and summarize completed scenario-sharded runs"
    )
    study_analysis.add_argument("config", type=Path, help="Completed study directory")
    study_analysis.add_argument("--output-dir", type=Path, required=True)
    calibration = commands.add_parser(
        "calibrate-integrative", help="Approximate global Gaussian test of frozen joint scores"
    )
    calibration.add_argument("config", type=Path, help="Completed integrative run directory")
    calibration.add_argument("--calibration-config", type=Path, required=True)
    calibration.add_argument("--output-dir", type=Path, required=True)
    score_study = commands.add_parser(
        "benchmark-score-calibration", help="Validate the conditional Gaussian score test"
    )
    score_study.add_argument("config", type=Path)
    score_study.add_argument(
        "--scenario",
        required=True,
        choices=("spherical_gaussian", "correlated_gaussian", "separated_mixture", "heavy_tailed"),
    )
    score_study.add_argument("--output-dir", type=Path, required=True)
    calibration_analysis = commands.add_parser(
        "analyze-score-calibration", help="Audit a completed four-scenario score study"
    )
    calibration_analysis.add_argument(
        "config", type=Path, help="Directory containing scenario shards"
    )
    calibration_analysis.add_argument("--output-dir", type=Path, required=True)
    raw_calibration = commands.add_parser(
        "calibrate-reference-pipeline", help="Test raw data against a declared simulated population"
    )
    raw_calibration.add_argument("config", type=Path, help="Observed dataset JSON")
    raw_calibration.add_argument("--calibration-config", type=Path, required=True)
    raw_calibration.add_argument("--output-dir", type=Path, required=True)
    pipeline_study = commands.add_parser(
        "benchmark-pipeline-calibration", help="Validate full-pipeline reference-law tests"
    )
    pipeline_study.add_argument("config", type=Path)
    pipeline_study.add_argument(
        "--case",
        required=True,
        choices=("matched_null", "matched_complementary", "grouped_null", "grouped_complementary"),
    )
    pipeline_study.add_argument("--output-dir", type=Path, required=True)
    pipeline_analysis = commands.add_parser(
        "analyze-pipeline-calibration", help="Audit a completed raw-reference calibration study"
    )
    pipeline_analysis.add_argument("config", type=Path)
    pipeline_analysis.add_argument("--output-dir", type=Path, required=True)
    empirical = commands.add_parser(
        "calibrate-empirical-integrative",
        help="Approximate fitted Gaussian test of continuous raw views",
    )
    empirical.add_argument("config", type=Path)
    empirical.add_argument("--calibration-config", type=Path, required=True)
    empirical.add_argument("--output-dir", type=Path, required=True)
    empirical_study = commands.add_parser(
        "benchmark-empirical-calibration", help="Validate fitted raw-Gaussian tests"
    )
    empirical_study.add_argument("config", type=Path)
    empirical_study.add_argument(
        "--case",
        required=True,
        choices=("spherical_gaussian", "correlated_gaussian", "complementary", "heavy_tailed"),
    )
    empirical_study.add_argument("--output-dir", type=Path, required=True)
    empirical_audit = commands.add_parser(
        "analyze-empirical-calibration", help="Audit completed empirical calibration cases"
    )
    empirical_audit.add_argument("config", type=Path)
    empirical_audit.add_argument("--output-dir", type=Path, required=True)
    from super_deli.models.joint_config import CASES as JOINT_CASES

    for command, help_text in (
        ("calibrate-joint-integrative", "Fit a joint genomic-phenotypic raw null"),
        ("benchmark-joint-calibration", "Validate joint empirical nulls"),
        ("analyze-joint-calibration", "Audit joint calibration studies"),
    ):
        joint_parser = commands.add_parser(command, help=help_text)
        joint_parser.add_argument("config", type=Path)
        joint_parser.add_argument("--output-dir", type=Path, required=True)
        if command == "calibrate-joint-integrative":
            joint_parser.add_argument("--calibration-config", type=Path, required=True)
        elif command == "benchmark-joint-calibration":
            joint_parser.add_argument("--case", required=True, choices=JOINT_CASES)
    linked = commands.add_parser(
        "calibrate-linked-genomic", help="Fit an empirical whole-block genotype null"
    )
    linked.add_argument("config", type=Path)
    linked.add_argument("--calibration-config", type=Path, required=True)
    linked.add_argument("--output-dir", type=Path, required=True)
    linked_study = commands.add_parser(
        "benchmark-linked-calibration", help="Validate linkage-aware empirical genomic nulls"
    )
    linked_study.add_argument("config", type=Path)
    from super_deli.models.linked_config import CASES as LINKED_CASES

    linked_study.add_argument("--case", required=True, choices=LINKED_CASES)
    linked_study.add_argument("--output-dir", type=Path, required=True)
    linked_audit = commands.add_parser(
        "analyze-linked-calibration", help="Audit completed linkage-aware empirical genomic cases"
    )
    linked_audit.add_argument("config", type=Path)
    linked_audit.add_argument("--output-dir", type=Path, required=True)
    compare_genomic = commands.add_parser(
        "compare-genomic-calibration",
        help="Compare byte-paired pooled-HWE and block-profile outcomes",
    )
    compare_genomic.add_argument("config", type=Path, help="Legacy pooled-HWE study directory")
    compare_genomic.add_argument("--current-study", type=Path, required=True)
    compare_genomic.add_argument("--output-dir", type=Path, required=True)
    genomic = commands.add_parser(
        "calibrate-genomic", help="Fit a pooled unlinked diploid HWE null"
    )
    genomic.add_argument("config", type=Path)
    genomic.add_argument("--calibration-config", type=Path, required=True)
    genomic.add_argument("--output-dir", type=Path, required=True)
    genomic_study = commands.add_parser(
        "benchmark-genomic-calibration", help="Validate fitted genomic null tests"
    )
    genomic_study.add_argument("config", type=Path)
    from super_deli.models.genomic_config import CASES as GENOMIC_CASES

    genomic_study.add_argument("--case", required=True, choices=GENOMIC_CASES)
    genomic_study.add_argument("--output-dir", type=Path, required=True)
    genomic_audit = commands.add_parser(
        "analyze-genomic-calibration", help="Audit completed genomic fitted-null cases"
    )
    genomic_audit.add_argument("config", type=Path)
    genomic_audit.add_argument("--output-dir", type=Path, required=True)
    grouped = commands.add_parser(
        "calibrate-grouped-integrative", help="Fit a hierarchical continuous-data null"
    )
    grouped.add_argument("config", type=Path)
    grouped.add_argument("--calibration-config", type=Path, required=True)
    grouped.add_argument("--output-dir", type=Path, required=True)
    grouped_study = commands.add_parser(
        "benchmark-grouped-calibration", help="Validate unequal-sampling fitted nulls"
    )
    grouped_study.add_argument("config", type=Path)
    grouped_study.add_argument(
        "--case",
        required=True,
        choices=(
            "independent_gaussian",
            "correlated_gaussian",
            "complementary",
            "heavy_tailed",
            "zero_conditional_gaussian",
            "rank_two_gaussian",
            "weak_conditional_gaussian",
        ),
    )
    grouped_study.add_argument("--output-dir", type=Path, required=True)
    grouped_audit = commands.add_parser(
        "analyze-grouped-calibration", help="Audit a grouped fitted-null study"
    )
    grouped_audit.add_argument("config", type=Path)
    grouped_audit.add_argument("--output-dir", type=Path, required=True)
    comparison = commands.add_parser(
        "compare-grouped-calibration", help="Compare byte-identical grouped observed cohorts"
    )
    comparison.add_argument("config", type=Path, help="Legacy schema-1 study directory")
    comparison.add_argument("--current-study", type=Path, required=True)
    comparison.add_argument("--output-dir", type=Path, required=True)
    baseline = commands.add_parser(
        "baseline-genomic", help="Fit candidate PCA/GMM genetic clusters"
    )
    baseline.add_argument("config", type=Path, help="Versioned dataset JSON configuration")
    baseline.add_argument(
        "--model-config", type=Path, required=True, help="Baseline assumptions and settings JSON"
    )
    baseline.add_argument(
        "--blocks", type=Path, help="One block ID per original source locus in JSON"
    )
    baseline.add_argument("--modality", help="Select exactly one declared genomic modality")
    baseline.add_argument(
        "--output-dir", type=Path, required=True, help="New analysis output directory"
    )
    recovery = commands.add_parser(
        "diagnose-integrative-recovery", help="Run the frozen four-group point-diagnostic grid"
    )
    recovery.add_argument("config", type=Path)
    recovery.add_argument("--source-artifacts", type=Path, required=True)
    recovery.add_argument("--output-dir", type=Path, required=True)
    recovery.add_argument("--workers", type=int, default=1)
    recovery_audit = commands.add_parser(
        "analyze-integrative-recovery", help="Audit recovery diagnostics without refitting models"
    )
    recovery_audit.add_argument("config", type=Path)
    recovery_audit.add_argument("--output-dir", type=Path, required=True)
    grouped_pca = commands.add_parser(
        "benchmark-grouped-pca", help="Run the frozen grouped PCA fit-level comparison"
    )
    grouped_pca.add_argument("config", type=Path)
    grouped_pca.add_argument("--source-artifacts", type=Path, required=True)
    grouped_pca.add_argument("--output-dir", type=Path, required=True)
    grouped_pca.add_argument("--workers", type=int, default=1)
    grouped_pca_audit = commands.add_parser(
        "analyze-grouped-pca", help="Audit grouped PCA recovery and stability without refitting"
    )
    grouped_pca_audit.add_argument("config", type=Path)
    grouped_pca_audit.add_argument("--output-dir", type=Path, required=True)
    grouped_recovery = commands.add_parser(
        "diagnose-grouped-recovery", help="Run the frozen grouped four-group diagnostic grid"
    )
    grouped_recovery.add_argument("config", type=Path)
    grouped_recovery.add_argument("--source-artifacts", type=Path, required=True)
    grouped_recovery.add_argument("--output-dir", type=Path, required=True)
    grouped_recovery.add_argument("--workers", type=int, default=1)
    grouped_recovery_audit = commands.add_parser(
        "analyze-grouped-recovery", help="Audit grouped recovery without model refits"
    )
    grouped_recovery_audit.add_argument("config", type=Path)
    grouped_recovery_audit.add_argument("--output-dir", type=Path, required=True)
    full_diagnostic = commands.add_parser(
        "diagnose-full-grouped",
        help="Diagnose full-covariance grouped recovery without calibration",
    )
    full_diagnostic.add_argument("config", type=Path)
    full_diagnostic.add_argument("--source-artifacts", type=Path, required=True)
    full_diagnostic.add_argument("--output-dir", type=Path, required=True)
    full_diagnostic.add_argument("--workers", type=int, default=1)
    full_audit = commands.add_parser(
        "analyze-full-grouped", help="Audit full grouped diagnostics without refitting"
    )
    full_audit.add_argument("config", type=Path)
    full_audit.add_argument("--output-dir", type=Path, required=True)
    likelihood = commands.add_parser(
        "calibrate-grouped-likelihood", help="Calibrate full-pipeline grouped likelihood gain"
    )
    likelihood.add_argument("config", type=Path)
    likelihood.add_argument("--calibration-config", type=Path, required=True)
    likelihood.add_argument("--output-dir", type=Path, required=True)
    likelihood_study = commands.add_parser(
        "benchmark-grouped-likelihood", help="Validate grouped AIC and likelihood calibration"
    )
    likelihood_study.add_argument("config", type=Path)
    likelihood_study.add_argument("--case", required=True)
    likelihood_study.add_argument("--output-dir", type=Path, required=True)
    likelihood_audit = commands.add_parser(
        "analyze-grouped-likelihood", help="Audit all candidate densities without refitting"
    )
    likelihood_audit.add_argument("config", type=Path)
    likelihood_audit.add_argument("--output-dir", type=Path, required=True)
    boundary_diagnostic = commands.add_parser(
        "diagnose-grouped-boundaries", help="Compare fitted and known Gaussian simulation laws"
    )
    boundary_diagnostic.add_argument("config", type=Path)
    boundary_diagnostic.add_argument("--case", required=True)
    boundary_diagnostic.add_argument("--output-dir", type=Path, required=True)
    boundary_audit = commands.add_parser(
        "analyze-grouped-boundaries", help="Audit paired boundary diagnostics without fitting"
    )
    boundary_audit.add_argument("config", type=Path)
    boundary_audit.add_argument("--archived-study", type=Path)
    boundary_audit.add_argument("--output-dir", type=Path, required=True)
    shared = commands.add_parser(
        "benchmark-shared-covariance", help="Run a frozen paired integrative covariance study"
    )
    shared.add_argument("config", type=Path)
    shared.add_argument("--source-artifacts", type=Path, required=True)
    shared.add_argument("--output-dir", type=Path, required=True)
    shared.add_argument("--workers", type=int, default=1)
    shared_audit = commands.add_parser(
        "analyze-shared-covariance", help="Audit the paired study and bootstrap support"
    )
    shared_audit.add_argument("config", type=Path)
    shared_audit.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        source = args.config.absolute()
        if args.command == "diagnose-full-grouped":
            from super_deli.benchmarking.full_grouped_diagnostics import (
                run_full_grouped_diagnostics,
            )
            from super_deli.models.full_grouped_diagnostic_config import FullGroupedDiagnosticConfig

            with redirect_stdout(sys.stderr):
                summary = run_full_grouped_diagnostics(
                    FullGroupedDiagnosticConfig.from_dict(read_json(source)),
                    source_artifacts=args.source_artifacts,
                    output_dir=args.output_dir,
                    workers=args.workers,
                )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "analyze-full-grouped":
            from super_deli.benchmarking.full_grouped_diagnostic_analysis import (
                audit_full_grouped_diagnostics,
            )

            summary = audit_full_grouped_diagnostics(source, output_dir=args.output_dir)
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "diagnose-grouped-boundaries":
            from super_deli.benchmarking.grouped_boundary_diagnostics import run_case
            from super_deli.models.grouped_likelihood_config import GroupedLikelihoodStudyConfig

            with redirect_stdout(sys.stderr):
                summary = run_case(
                    GroupedLikelihoodStudyConfig.from_dict(read_json(source)),
                    case=args.case,
                    output_dir=args.output_dir,
                )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "analyze-grouped-boundaries":
            from super_deli.benchmarking.grouped_boundary_analysis import audit_study

            summary = audit_study(
                source, output_dir=args.output_dir, archived_study=args.archived_study
            )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "calibrate-grouped-likelihood":
            from super_deli.models.grouped_likelihood_config import GroupedLikelihoodConfig
            from super_deli.workflows.grouped_likelihood import run_grouped_likelihood

            with redirect_stdout(sys.stderr):
                summary = run_grouped_likelihood(
                    load_config(source),
                    config=GroupedLikelihoodConfig.from_dict(read_json(args.calibration_config)),
                    base_dir=source.parent,
                    output_dir=args.output_dir,
                )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "benchmark-grouped-likelihood":
            from super_deli.benchmarking.grouped_likelihood import run_grouped_likelihood_study
            from super_deli.models.grouped_likelihood_config import GroupedLikelihoodStudyConfig

            with redirect_stdout(sys.stderr):
                summary = run_grouped_likelihood_study(
                    GroupedLikelihoodStudyConfig.from_dict(read_json(source)),
                    case=args.case,
                    output_dir=args.output_dir,
                )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "analyze-grouped-likelihood":
            from super_deli.benchmarking.grouped_likelihood_analysis import audit_grouped_likelihood

            summary = audit_grouped_likelihood(source, output_dir=args.output_dir)
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "benchmark-grouped-pca":
            from super_deli.benchmarking.grouped_pca_study import run_grouped_pca_study
            from super_deli.models.grouped_pca_study_config import GroupedPCAStudyConfig

            with redirect_stdout(sys.stderr):
                summary = run_grouped_pca_study(
                    GroupedPCAStudyConfig.from_dict(read_json(source)),
                    source_artifacts=args.source_artifacts,
                    output_dir=args.output_dir,
                    workers=args.workers,
                )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "analyze-grouped-pca":
            from super_deli.benchmarking.grouped_pca_analysis import audit_grouped_pca_study

            summary = audit_grouped_pca_study(source, output_dir=args.output_dir)
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "diagnose-grouped-recovery":
            from super_deli.benchmarking.grouped_recovery import run_grouped_recovery
            from super_deli.models.grouped_recovery_config import GroupedRecoveryConfig

            with redirect_stdout(sys.stderr):
                summary = run_grouped_recovery(
                    GroupedRecoveryConfig.from_dict(read_json(source)),
                    source_artifacts=args.source_artifacts,
                    output_dir=args.output_dir,
                    workers=args.workers,
                )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "analyze-grouped-recovery":
            from super_deli.benchmarking.grouped_recovery_analysis import audit_grouped_recovery

            summary = audit_grouped_recovery(source, output_dir=args.output_dir)
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "benchmark-shared-covariance":
            from super_deli.benchmarking.shared_study import run_shared_study
            from super_deli.models.shared_config import SharedStudyConfig

            with redirect_stdout(sys.stderr):
                summary = run_shared_study(
                    SharedStudyConfig.from_dict(read_json(source)),
                    source_artifacts=args.source_artifacts,
                    output_dir=args.output_dir,
                    workers=args.workers,
                )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "analyze-shared-covariance":
            from super_deli.benchmarking.shared_analysis import audit_shared_study

            summary = audit_shared_study(source, output_dir=args.output_dir)
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "diagnose-integrative-recovery":
            from super_deli.benchmarking.recovery import run_recovery
            from super_deli.models.recovery_config import RecoveryConfig

            with redirect_stdout(sys.stderr):
                summary = run_recovery(
                    RecoveryConfig.from_dict(read_json(source)),
                    source_artifacts=args.source_artifacts,
                    output_dir=args.output_dir,
                    workers=args.workers,
                )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "analyze-integrative-recovery":
            from super_deli.benchmarking.recovery_analysis import audit_recovery

            summary = audit_recovery(source, output_dir=args.output_dir)
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "compare-grouped-calibration":
            from super_deli.benchmarking.grouped_comparison import compare_grouped_studies

            summary = compare_grouped_studies(
                source, args.current_study, output_dir=args.output_dir
            )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command in {
            "calibrate-joint-integrative",
            "benchmark-joint-calibration",
            "analyze-joint-calibration",
        }:
            with redirect_stdout(sys.stderr):
                if args.command == "calibrate-joint-integrative":
                    from super_deli.models.joint_config import JointCalibrationConfig
                    from super_deli.workflows.joint_calibration import run_joint_calibration

                    summary = run_joint_calibration(
                        load_config(source),
                        config=JointCalibrationConfig.from_dict(read_json(args.calibration_config)),
                        base_dir=source.parent,
                        output_dir=args.output_dir,
                    )
                elif args.command == "benchmark-joint-calibration":
                    from super_deli.benchmarking.joint_study import run_joint_study
                    from super_deli.models.joint_config import JointStudyConfig

                    summary = run_joint_study(
                        JointStudyConfig.from_dict(read_json(source)),
                        case=args.case,
                        output_dir=args.output_dir,
                    )
                else:
                    from super_deli.benchmarking.joint_analysis import audit_joint_study

                    summary = audit_joint_study(source, output_dir=args.output_dir)
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return (
                0
                if "inference" not in summary or summary["inference"]["status"] == "available"
                else 1
            )
        if args.command == "calibrate-linked-genomic":
            from super_deli.models.linked_config import LinkedCalibrationConfig
            from super_deli.workflows.linked_calibration import run_linked_calibration

            with redirect_stdout(sys.stderr):
                summary = run_linked_calibration(
                    load_config(source),
                    config=LinkedCalibrationConfig.from_dict(read_json(args.calibration_config)),
                    base_dir=source.parent,
                    output_dir=args.output_dir,
                )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0 if summary["inference"]["status"] == "available" else 1
        if args.command == "benchmark-linked-calibration":
            from super_deli.benchmarking.linked_study import run_linked_study
            from super_deli.models.linked_config import LinkedStudyConfig

            with redirect_stdout(sys.stderr):
                summary = run_linked_study(
                    LinkedStudyConfig.from_dict(read_json(source)),
                    case=args.case,
                    output_dir=args.output_dir,
                )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "analyze-linked-calibration":
            from super_deli.benchmarking.linked_analysis import audit_linked_study

            summary = audit_linked_study(source, output_dir=args.output_dir)
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "compare-genomic-calibration":
            from super_deli.benchmarking.genomic_comparison import compare_genomic_studies

            summary = compare_genomic_studies(
                source, args.current_study, output_dir=args.output_dir
            )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "calibrate-genomic":
            from super_deli.models.genomic_config import GenomicCalibrationConfig
            from super_deli.workflows.genomic_calibration import run_genomic_calibration

            with redirect_stdout(sys.stderr):
                summary = run_genomic_calibration(
                    load_config(source),
                    config=GenomicCalibrationConfig.from_dict(read_json(args.calibration_config)),
                    base_dir=source.parent,
                    output_dir=args.output_dir,
                )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0 if summary["inference"]["status"] == "available" else 1
        if args.command == "benchmark-genomic-calibration":
            from super_deli.benchmarking.genomic_study import run_genomic_study
            from super_deli.models.genomic_config import GenomicStudyConfig

            with redirect_stdout(sys.stderr):
                summary = run_genomic_study(
                    GenomicStudyConfig.from_dict(read_json(source)),
                    case=args.case,
                    output_dir=args.output_dir,
                )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "analyze-genomic-calibration":
            from super_deli.benchmarking.genomic_analysis import audit_genomic_study

            summary = audit_genomic_study(source, output_dir=args.output_dir)
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "calibrate-grouped-integrative":
            from super_deli.models.grouped_config import GroupedCalibrationConfig
            from super_deli.workflows.grouped_calibration import run_grouped_calibration

            summary = run_grouped_calibration(
                load_config(source),
                config=GroupedCalibrationConfig.from_dict(read_json(args.calibration_config)),
                base_dir=source.parent,
                output_dir=args.output_dir,
            )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0 if summary["inference"]["status"] == "available" else 1
        if args.command == "benchmark-grouped-calibration":
            from super_deli.benchmarking.grouped_study import run_grouped_study
            from super_deli.models.grouped_config import GroupedStudyConfig

            summary = run_grouped_study(
                GroupedStudyConfig.from_dict(read_json(source)),
                case=args.case,
                output_dir=args.output_dir,
            )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "analyze-grouped-calibration":
            from super_deli.benchmarking.grouped_analysis import audit_grouped_study

            summary = audit_grouped_study(source, output_dir=args.output_dir)
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "calibrate-empirical-integrative":
            from super_deli.models.empirical_config import EmpiricalCalibrationConfig
            from super_deli.workflows.empirical_calibration import run_empirical_calibration

            summary = run_empirical_calibration(
                load_config(source),
                config=EmpiricalCalibrationConfig.from_dict(read_json(args.calibration_config)),
                base_dir=source.parent,
                output_dir=args.output_dir,
            )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0 if summary["inference"]["status"] == "available" else 1
        if args.command == "benchmark-empirical-calibration":
            from super_deli.benchmarking.empirical_study import run_empirical_study
            from super_deli.models.empirical_config import EmpiricalStudyConfig

            summary = run_empirical_study(
                EmpiricalStudyConfig.from_dict(read_json(source)),
                case=args.case,
                output_dir=args.output_dir,
            )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "analyze-empirical-calibration":
            from super_deli.benchmarking.empirical_analysis import audit_empirical_study

            summary = audit_empirical_study(source, output_dir=args.output_dir)
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "analyze-pipeline-calibration":
            from super_deli.benchmarking.pipeline_analysis import audit_pipeline_study

            summary = audit_pipeline_study(source, output_dir=args.output_dir)
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "calibrate-reference-pipeline":
            from super_deli.models.pipeline_config import PipelineCalibrationConfig
            from super_deli.workflows.pipeline_calibration import run_reference_calibration

            with redirect_stdout(sys.stderr):
                summary = run_reference_calibration(
                    load_config(source),
                    config=PipelineCalibrationConfig.from_dict(read_json(args.calibration_config)),
                    base_dir=source.parent,
                    output_dir=args.output_dir,
                )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0 if summary["inference"]["status"] == "available" else 1
        if args.command == "benchmark-pipeline-calibration":
            from super_deli.benchmarking.pipeline_study import run_pipeline_study
            from super_deli.models.pipeline_config import PipelineStudyConfig

            with redirect_stdout(sys.stderr):
                summary = run_pipeline_study(
                    PipelineStudyConfig.from_dict(read_json(source)),
                    case=args.case,
                    output_dir=args.output_dir,
                )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "calibrate-integrative":
            from super_deli.models.calibration_config import GaussianCalibrationConfig
            from super_deli.workflows.score_calibration import run_integrative_calibration

            summary = run_integrative_calibration(
                source,
                config=GaussianCalibrationConfig.from_dict(read_json(args.calibration_config)),
                output_dir=args.output_dir,
            )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0 if summary["inference"]["status"] == "available" else 1
        if args.command == "benchmark-score-calibration":
            from super_deli.benchmarking.score_calibration import run_score_study

            summary = run_score_study(
                read_json(source), output_dir=args.output_dir, scenario=args.scenario
            )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "analyze-score-calibration":
            from super_deli.benchmarking.calibration_analysis import audit_score_study

            summary = audit_score_study(source, output_dir=args.output_dir)
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "analyze-integrative-study":
            from super_deli.benchmarking.study_analysis import write_analysis

            summary = write_analysis(source, output_dir=args.output_dir)
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "benchmark-integrative":
            from super_deli.benchmarking.config import BenchmarkConfig
            from super_deli.benchmarking.runner import run_benchmark

            with redirect_stdout(sys.stderr):
                summary = run_benchmark(
                    BenchmarkConfig.from_dict(read_json(source)), output_dir=args.output_dir
                )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        config = load_config(source)
        if args.command == "cluster-integrative":
            from super_deli.models.integrative_config import IntegrativeConfig
            from super_deli.workflows.integrative import run_integrative

            with redirect_stdout(sys.stderr):
                summary = run_integrative(
                    config,
                    model_config=IntegrativeConfig.from_dict(read_json(args.model_config)),
                    base_dir=source.parent,
                    output_dir=args.output_dir,
                )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "cluster-tabular":
            from super_deli.models.tabular_cluster_config import TabularClusterConfig
            from super_deli.workflows.tabular_gmm import run_tabular_gmm

            summary = run_tabular_gmm(
                config,
                model_config=TabularClusterConfig.from_dict(read_json(args.model_config)),
                base_dir=source.parent,
                output_dir=args.output_dir,
                modality_name=args.modality,
            )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "cluster-shape":
            from super_deli.models.shape_cluster_config import ShapeClusterConfig
            from super_deli.workflows.shape_gmm import run_shape_gmm

            summary = run_shape_gmm(
                config,
                model_config=ShapeClusterConfig.from_dict(read_json(args.model_config)),
                base_dir=source.parent,
                output_dir=args.output_dir,
                modality_name=args.modality,
            )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "shape-pca":
            from super_deli.models.shape_config import ShapeConfig
            from super_deli.workflows.shape_pca import run_shape_pca

            shape_data = run_shape_pca(
                config,
                model_config=ShapeConfig.from_dict(read_json(args.model_config)),
                base_dir=source.parent,
                output_dir=args.output_dir,
                modality_name=args.modality,
                reference_dir=args.reference_run,
            )
            print(json.dumps(shape_data.summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "detect-landmarks":
            from super_deli.models.marker_config import MarkerDetectorConfig
            from super_deli.workflows.landmark_detection import run_landmark_detection

            loaded = run_landmark_detection(
                config,
                detector_config=MarkerDetectorConfig.from_dict(read_json(args.detector_config)),
                base_dir=source.parent,
                output_dir=args.output_dir,
                modality_name=args.modality,
            )
            print(json.dumps(loaded.summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "ingest-images":
            from super_deli.io.images import ingest_images

            loaded_images = ingest_images(
                config,
                base_dir=source.parent,
                output_dir=args.output_dir,
                modality_names=args.modality,
            )
            print(json.dumps(loaded_images.summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "ingest-morphometric":
            from super_deli.io.morphometric import ingest_morphometric

            loaded_shapes = ingest_morphometric(
                config,
                base_dir=source.parent,
                output_dir=args.output_dir,
                modality_names=args.modality,
            )
            print(json.dumps(loaded_shapes.summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "ingest-tabular":
            from super_deli.io.tabular import ingest_tabular

            loaded_tables = ingest_tabular(
                config,
                base_dir=source.parent,
                output_dir=args.output_dir,
                modality_names=args.modality,
            )
            print(json.dumps(loaded_tables.summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "baseline-genomic":
            from super_deli.models.config import BaselineConfig

            try:
                from super_deli.workflows.genomic_baseline import parse_blocks, run_genomic_baseline
            except ImportError as exc:
                raise RuntimeError(
                    "The genomic baseline requires: pip install 'super-deli[genomics,baseline]'"
                ) from exc
            model = BaselineConfig.from_dict(read_json(args.model_config))
            blocks = parse_blocks(read_json(args.blocks)) if args.blocks is not None else None
            with redirect_stdout(sys.stderr):
                summary = run_genomic_baseline(
                    config,
                    model_config=model,
                    base_dir=source.parent,
                    output_dir=args.output_dir,
                    block_ids=blocks,
                    modality_name=args.modality,
                )
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "ingest-genomic":
            from super_deli.io.genomic import ingest_genomic

            with redirect_stdout(sys.stderr):
                loaded = ingest_genomic(
                    config,
                    base_dir=source.parent,
                    output_dir=args.output_dir,
                    modality_name=args.modality,
                )
            print(json.dumps(loaded.summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        report = validate_dataset(config, base_dir=source.parent)
        document = report.to_dict()
        document["provenance"] = collect_provenance(
            config, base_dir=source.parent, hash_inputs=args.hash_inputs and report.valid
        )
        if args.hash_inputs and not report.valid:
            document["provenance"]["input_hash_status"] = "skipped_invalid_dataset"
        if args.report is not None:
            write_json(document, args.report)
        print(json.dumps(document, indent=2, sort_keys=True, allow_nan=False))
        return 0 if report.valid else 1
    except (ConfigError, OSError, ValueError, RuntimeError) as exc:
        print(f"super-deli: {exc}", file=sys.stderr)
        return 2
