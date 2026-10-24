"""Installed repeated-partition/oracle study and two independent audit paths."""

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
        "benchmark-predictive-recovery",
        Path(__file__).with_name("smoke.json"),
        "--output-dir",
        root / "study",
        "--workers",
        2,
    )
    cli(
        "analyze-predictive-recovery",
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
            str(Path(__file__).with_name("check_recovery.py")),
            "--study-root",
            str(root / "study"),
            "--audit-root",
            str(root / "audit"),
            "--output",
            str(root / "independent.json"),
        ],
        check=True,
    )
    summary = json.loads((root / "audit/summary.json").read_text())
    assert summary["successful_searches"] == 68 and summary["successful_oracles"] == 18
    assert summary["failed_searches"] == summary["failed_oracles"] == summary["withheld_parts"] == 0
    assert len(summary["primary_rates"]) == 16 and len(summary["paired"]) == 8
    print(
        json.dumps(
            {
                "predictive_recovery_smoke_passed": True,
                "candidate_searches": 68,
                "single_k_diagnostics": 18,
                "primary_rates": 16,
                "independent_checker": json.loads((root / "independent.json").read_text()),
            }
        )
    )


if __name__ == "__main__":
    main()
