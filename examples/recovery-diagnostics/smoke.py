"""Create a tiny current-runtime source archive, then exercise both installed CLIs."""

import argparse
import hashlib
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from super_deli.benchmarking import joint_study
from super_deli.benchmarking.recovery_inputs import inventory
from super_deli.io.config import read_json, write_json
from super_deli.models.joint_config import JointStudyConfig
from super_deli.models.recovery_config import RecoveryConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-study", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(exist_ok=False)
    source = root / "source"
    (source / "study").mkdir(parents=True)
    study = JointStudyConfig.from_dict(read_json(args.joint_study))
    study = replace(
        study,
        n_datasets=1,
        n_units=24,
        calibration=replace(
            study.calibration, n_simulations=1, model=replace(study.calibration.model, max_k=4)
        ),
    )
    write_json(study.to_dict(), source / "study.json")
    joint_study.run_joint_study(
        study, case="complementary", output_dir=source / "study/complementary"
    )
    (source / "study-analysis").mkdir()
    write_json(inventory(source / "study"), source / "study-analysis/files.json")
    write_json(
        {
            "super_deli/benchmarking/joint_study.py": hashlib.sha256(
                Path(joint_study.__file__).read_bytes()
            ).hexdigest()
        },
        source / "runtime_sources.json",
    )

    def digest(name):
        return hashlib.sha256((source / name).read_bytes()).hexdigest()

    config = RecoveryConfig(
        digest("study.json"),
        digest("study-analysis/files.json"),
        digest("runtime_sources.json"),
        n_datasets=1,
        small_n=24,
        large_n=32,
    )
    write_json(config.to_dict(), root / "diagnostic.json")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "super_deli",
            "diagnose-integrative-recovery",
            str(root / "diagnostic.json"),
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
            "analyze-integrative-recovery",
            str(root / "study"),
            "--output-dir",
            str(root / "audit"),
        ],
        check=True,
    )
    summary = read_json(root / "audit/summary.json")
    if summary["successful_points"] != 21 or summary["successful_truth_initialized"] != 3:
        raise RuntimeError("Installed diagnostic smoke failed its complete-fit accounting")


if __name__ == "__main__":
    main()
