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
from super_deli.io.config import load_config, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and ingest Super Deli datasets")
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
    args = parser.parse_args(argv)
    try:
        source = args.config.absolute()
        config = load_config(source)
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
