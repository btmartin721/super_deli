"""Run installed paired boundary diagnostics and their independent audit."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from super_deli.benchmarking.grouped_boundary_reference import CASES
from super_deli.io.config import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(exist_ok=False)
    (root / "study").mkdir()
    for case in CASES:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "super_deli",
                "diagnose-grouped-boundaries",
                str(args.study.resolve()),
                "--case",
                case,
                "--output-dir",
                str(root / "study" / case),
            ],
            check=True,
        )
    (root / "archived").mkdir()
    legacy = read_json(args.study)
    legacy["random_seed"] = 20261115
    legacy["calibration"]["random_seed"] = 20261115
    legacy_path = root / "archived-config.json"
    legacy_path.write_text(json.dumps(legacy))
    for case in CASES:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "super_deli",
                "benchmark-grouped-likelihood",
                str(legacy_path),
                "--case",
                case,
                "--output-dir",
                str(root / "archived" / case),
            ],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "super_deli",
            "analyze-grouped-boundaries",
            str(root / "study"),
            "--archived-study",
            str(root / "archived"),
            "--output-dir",
            str(root / "audit"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("check_tails.py")),
            "--study-root",
            str(root / "study"),
            "--archived-study",
            str(root / "archived"),
            "--tails",
            str(root / "audit/tails.csv"),
            "--output",
            str(root / "independent.json"),
        ],
        check=True,
    )
    result = read_json(root / "audit/summary.json")
    assert result["new_unique_successful_fits"] == 15
    assert result["unique_pipeline_records_audited"] == 25


if __name__ == "__main__":
    main()
