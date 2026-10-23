"""Exercise installed grouped likelihood study, direct calibration and no-refit audit."""

import argparse
import subprocess
import sys
from pathlib import Path

from super_deli.io.config import read_json
from super_deli.models.grouped_likelihood_config import CASES


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
                "benchmark-grouped-likelihood",
                str(args.study.resolve()),
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
            "analyze-grouped-likelihood",
            str(root / "study"),
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
            "--paired-table",
            str(root / "audit/paired.csv"),
            "--output",
            str(root / "independent-tails.json"),
        ],
        check=True,
    )
    dataset = root / "study/independent_gaussian/dataset_0000"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "super_deli",
            "calibrate-grouped-likelihood",
            str(dataset / "observed/dataset.json"),
            "--calibration-config",
            str(dataset / "calibration/configuration.json"),
            "--output-dir",
            str(root / "direct"),
        ],
        check=True,
    )
    result = read_json(root / "audit/summary.json")
    if result["successful_fits"] != 20 or result["datasets"] != 10:
        raise RuntimeError("Installed smoke did not complete the planned study fits")


if __name__ == "__main__":
    main()
