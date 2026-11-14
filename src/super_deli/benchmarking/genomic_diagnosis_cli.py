"""Lazy registration of a fit-free, source-bound archival diagnosis."""

from pathlib import Path


def register(commands):
    parser = commands.add_parser(
        "diagnose-genomic-neural", help="Fit-free audit of archived genomic neural control splits"
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument("--diagnosis-config", type=Path, required=True)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--output-dir", type=Path, required=True)


def dispatch(args):
    from super_deli.benchmarking.genomic_diagnosis import run_genomic_diagnosis
    from super_deli.benchmarking.genomic_diagnosis_config import GenomicDiagnosisConfig
    from super_deli.io.config import read_json

    return run_genomic_diagnosis(
        args.archive,
        config=GenomicDiagnosisConfig.from_dict(read_json(args.diagnosis_config)),
        output_dir=args.output_dir,
        case_ids=args.case_id,
    )
