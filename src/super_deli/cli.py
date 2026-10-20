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
    args = parser.parse_args(argv)
    try:
        source = args.config.absolute()
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
