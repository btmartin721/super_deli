"""Exercise installed shared-covariance study/audit CLIs on a tiny prior diagnostic."""

import argparse
import subprocess
import sys
from pathlib import Path

from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json
from super_deli.models.joint_config import JointStudyConfig
from super_deli.models.shared_config import SharedStudyConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifacts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_artifacts.resolve()
    root = args.output_dir.resolve()
    root.mkdir(exist_ok=False)
    prior = JointStudyConfig.from_dict(read_json(source / "study/source_study.json"))
    config = SharedStudyConfig(
        prior.calibration.model,
        fingerprint_file(source / "study/files.json").sha256,
        fingerprint_file(source / "study/summary.json").sha256,
        fingerprint_file(source / "study/source_study.json").sha256,
        n_datasets=1,
        small_n=24,
        large_n=32,
        stability_datasets=1,
        stability_draws=2,
    )
    write_json(config.to_dict(), root / "configuration.json")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "super_deli",
            "benchmark-shared-covariance",
            str(root / "configuration.json"),
            "--source-artifacts",
            str(source),
            "--output-dir",
            str(root / "study"),
            "--workers",
            "2",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "super_deli",
            "analyze-shared-covariance",
            str(root / "study"),
            "--output-dir",
            str(root / "audit"),
        ],
        check=True,
    )
    result = read_json(root / "audit/summary.json")
    if result["successful_points"] != 12 or result["successful_bootstrap"] != 8:
        raise RuntimeError("Installed paired study did not complete its planned fits")


if __name__ == "__main__":
    main()
