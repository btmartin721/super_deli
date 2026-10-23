"""Reuse a freshly generated diagnostic fixture and exercise the installed study CLIs."""

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import super_deli
from super_deli.io.config import read_json, write_json
from super_deli.models.grouped_config import GroupedStudyConfig
from super_deli.models.grouped_pca_study_config import PINS, GroupedPCAStudyConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifacts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(exist_ok=False)
    # The existing fixture is read-only. Add runtime provenance to an owned copy.
    source = root / "source"
    shutil.copytree(args.source_artifacts, source)
    package = Path(super_deli.__file__).resolve().parent
    if not (source / "runtime_sources.json").exists():
        write_json(
            {
                p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(package.rglob("*.py"))
            },
            source / "runtime_sources.json",
        )
    prior = GroupedStudyConfig.from_dict(read_json(source / "study/source_study.json"))
    options = {
        field: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for field, name in PINS.items()
    }
    config = GroupedPCAStudyConfig(
        model=prior.calibration.model,
        **options,
        n_datasets=prior.n_datasets,
        small_n=prior.n_units,
        large_n=prior.n_units + 8,
        n_stability=1,
        n_bootstrap=2,
        random_seed=20261107,
    )
    write_json(config.to_dict(), root / "study.json")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "super_deli",
            "benchmark-grouped-pca",
            str(root / "study.json"),
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
            "analyze-grouped-pca",
            str(root / "study"),
            "--output-dir",
            str(root / "audit"),
        ],
        check=True,
    )
    result = read_json(root / "audit/summary.json")
    if (
        result["successful_points"] != 28
        or result["successful_bootstrap"] != 16
        or result["reproduced_archived"] != 1
    ):
        raise RuntimeError("Installed grouped PCA smoke failed its fit accounting")


if __name__ == "__main__":
    main()
