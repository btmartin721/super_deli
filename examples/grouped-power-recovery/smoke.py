"""Installed CLI smoke: compact retention, storage gate, audit and independent tails."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from super_deli.benchmarking.grouped_power_preflight import preflight_config
from super_deli.io.config import read_json, write_json
from super_deli.models.grouped_power_config import GroupedPowerStudyConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    configuration = Path(__file__).with_name("smoke.json")
    config = GroupedPowerStudyConfig.from_dict(read_json(configuration))
    write_json(preflight_config(config, random_seed=20261123).to_dict(), root / "probe.json")

    def cli(*arguments):
        subprocess.run([sys.executable, "-m", "super_deli", *map(str, arguments)], check=True)

    cli(
        "benchmark-grouped-power",
        root / "probe.json",
        "--output-dir",
        root / "probe",
        "--workers",
        2,
    )
    cli(
        "analyze-grouped-power",
        root / "probe",
        "--output-dir",
        root / "probe-audit",
        "--scratch",
        root / ".scratch",
        "--workers",
        2,
    )
    cli(
        "preflight-grouped-power",
        configuration,
        "--probe-study",
        root / "probe",
        "--probe-audit",
        root / "probe-audit",
        "--artifact-root",
        root,
        "--scratch",
        root / ".scratch",
        "--output-file",
        root / "storage-preflight.json",
    )
    cli(
        "benchmark-grouped-power",
        configuration,
        "--preflight",
        root / "storage-preflight.json",
        "--output-dir",
        root / "study",
        "--workers",
        2,
    )
    cli(
        "analyze-grouped-power",
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
            str(Path(__file__).with_name("check_tails.py")),
            "--study-root",
            str(root / "study"),
            "--audit-root",
            str(root / "audit"),
            "--output",
            str(root / "independent.json"),
        ],
        check=True,
    )
    result = read_json(root / "audit/summary.json")
    assert result["fit_accounting"]["successful_fits"] == 32
    assert result["fit_accounting"]["failed_fits"] == 0
    assert result["primary_family_size"] == len(result["primary_rates"]) == 18
    assert result["numerically_audited_observations"] == 10
    print(
        json.dumps(
            {
                "smoke_passed": True,
                "main_fits": 32,
                "probe_fits": 8,
                "independent": read_json(root / "independent.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
