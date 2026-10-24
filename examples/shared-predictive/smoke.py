"""Installed paired raw workflows with numerical and standard-library audits."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)

    def cli(*parts):
        subprocess.run([sys.executable, "-m", "super_deli", *map(str, parts)], check=True)

    cli(
        "benchmark-shared-predictive",
        Path(__file__).with_name("smoke.json"),
        "--output-dir",
        root / "study",
        "--workers",
        2,
    )
    cli(
        "analyze-shared-predictive",
        root / "study",
        "--output-dir",
        root / "audit",
        "--scratch",
        root / ".scratch",
        "--workers",
        2,
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("check_shared.py")),
            "--study-root",
            str(root / "study"),
            "--audit-root",
            str(root / "audit"),
            "--output",
            str(root / "independent.json"),
        ],
        check=True,
    )
    result = json.loads((root / "audit/summary.json").read_text())
    assert (
        result["successful_fits"] == 96
        and result["failed_fits"] == result["screening_withheld"] == 0
    )
    assert len(result["primary_rates"]) == 20 and len(result["paired"]) == 15
    print(
        json.dumps(
            {
                "shared_predictive_smoke_passed": True,
                "candidate_searches": 96,
                "primary_rates": 20,
                "paired_comparisons": 15,
            }
        )
    )


if __name__ == "__main__":
    main()
