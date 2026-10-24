"""Run installed larger-validation smoke and independent no-refit checks."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from super_deli.benchmarking.grouped_boundary_reference import CASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(exist_ok=False)
    (root / "study").mkdir()
    config = Path(__file__).with_name("smoke.json")
    for case in CASES:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "super_deli",
                "diagnose-grouped-boundaries",
                str(config),
                "--case",
                case,
                "--output-dir",
                str(root / "study" / case),
            ],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "super_deli",
            "analyze-grouped-false-positives",
            str(root / "study"),
            "--workers",
            "2",
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
            "--tails",
            str(root / "audit/tails.csv"),
            "--output",
            str(root / "independent.json"),
        ],
        check=True,
    )
    result = json.loads((root / "audit/summary.json").read_text())
    assert result["unique_pipeline_records_audited"] == 15
    assert result["fit_accounting"]["successful_fits"] == 15
    assert len(result["primary_rates"]) == 10


if __name__ == "__main__":
    main()
