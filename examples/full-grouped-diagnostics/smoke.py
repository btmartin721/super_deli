"""Create a seven-case source and exercise both installed diagnostic commands."""

import argparse
import hashlib
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from super_deli.benchmarking import grouped_study
from super_deli.benchmarking.recovery_inputs import inventory
from super_deli.io.config import read_json, write_json
from super_deli.models.full_grouped_diagnostic_config import FullGroupedDiagnosticConfig
from super_deli.models.grouped_config import GroupedStudyConfig, study_cases


def make_source(root, declaration):
    source = root / "source"
    (source / "study").mkdir(parents=True)
    study = GroupedStudyConfig.from_dict(read_json(declaration))
    study = replace(
        study,
        n_datasets=1,
        n_units=64,
        calibration=replace(
            study.calibration,
            n_simulations=1,
            model=replace(study.calibration.model, max_k=4, n_init=10),
        ),
    )
    write_json(study.to_dict(), source / "study.json")
    for case in study_cases(study):
        grouped_study.run_grouped_study(study, case=case, output_dir=source / "study" / case)
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
    digest = lambda name: hashlib.sha256((source / name).read_bytes()).hexdigest()  # noqa: E731
    config = FullGroupedDiagnosticConfig(
        digest("study.json"),
        digest("study-audit/files.json"),
        digest("runtime_sources.json"),
        n_datasets=1,
        small_n=64,
        large_n=128,
        random_seed=20261113,
    )
    return source, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grouped-study", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(exist_ok=False)
    source, config = make_source(root, args.grouped_study)
    write_json(config.to_dict(), root / "diagnostic.json")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "super_deli",
            "diagnose-full-grouped",
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
            "analyze-full-grouped",
            str(root / "study"),
            "--output-dir",
            str(root / "audit"),
        ],
        check=True,
    )
    result = read_json(root / "audit/summary.json")
    if (
        result["successful_points"],
        result["successful_truth_initialized"],
        result["archived_tests"],
    ) != (18, 9, 7):
        raise RuntimeError("Installed diagnostic smoke did not complete all planned fits")


if __name__ == "__main__":
    main()
