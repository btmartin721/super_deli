"""Installed point-only smoke, frozen held-out projection and independent archive audit."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)

    def cli(*values):
        subprocess.run([sys.executable, "-m", "super_deli", *map(str, values)], check=True)

    cli(
        "benchmark-aic-gaussian",
        Path(__file__).with_name("smoke.json"),
        "--output-dir",
        root / "study",
        "--workers",
        2,
    )
    cli(
        "analyze-aic-gaussian",
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
            str(Path(__file__).with_name("check_candidates.py")),
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
    assert result["point_fits"] == result["successful_fits"] == 20
    assert result["failed_fits"] == 0 and result["primary_family_size"] == 20
    assert result["calibration_draws"] == result["stability_draws"] == 0
    print(
        json.dumps(
            {
                "aic_diagnostic_smoke_passed": True,
                "point_searches": 20,
                "independent_check": json.loads((root / "independent.json").read_text()),
            }
        )
    )


if __name__ == "__main__":
    main()
