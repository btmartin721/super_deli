"""Create a tiny current-runtime source archive, then exercise both installed CLIs."""

import argparse
import hashlib
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from super_deli.benchmarking import grouped_study
from super_deli.benchmarking.recovery_inputs import inventory
from super_deli.io.config import read_json, write_json
from super_deli.models.grouped_config import GroupedStudyConfig
from super_deli.models.grouped_recovery_config import GroupedRecoveryConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grouped-study", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(exist_ok=False)
    source = root / "source"
    (source / "study").mkdir(parents=True)
    study = GroupedStudyConfig.from_dict(read_json(args.grouped_study))
    study = replace(
        study,
        n_datasets=1,
        n_units=24,
        calibration=replace(
            study.calibration,
            n_simulations=1,
            model=replace(study.calibration.model, max_k=4, n_init=10),
        ),
    )
    write_json(study.to_dict(), source / "study.json")
    grouped_study.run_grouped_study(
        study, case="complementary", output_dir=source / "study/complementary"
    )
    (source / "study-audit").mkdir()
    write_json(inventory(source / "study"), source / "study-audit/files.json")
    write_json(
        {
            "super_deli/benchmarking/grouped_study.py": hashlib.sha256(
                Path(grouped_study.__file__).read_bytes()
            ).hexdigest()
        },
        source / "runtime_sources.json",
    )

    def digest(name):
        return hashlib.sha256((source / name).read_bytes()).hexdigest()

    config = GroupedRecoveryConfig(
        digest("study.json"),
        digest("study-audit/files.json"),
        digest("runtime_sources.json"),
        n_datasets=1,
        random_seed=20261105,
        small_n=24,
        large_n=32,
    )
    write_json(config.to_dict(), root / "diagnostic.json")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "super_deli",
            "diagnose-grouped-recovery",
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
            "analyze-grouped-recovery",
            str(root / "study"),
            "--output-dir",
            str(root / "audit"),
        ],
        check=True,
    )
    summary = read_json(root / "audit/summary.json")
    if summary["successful_points"] != 30 or summary["successful_truth_initialized"] != 3:
        raise RuntimeError("Installed diagnostic smoke failed its complete-fit accounting")


if __name__ == "__main__":
    main()
