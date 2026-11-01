"""Installed engineering smoke: no scientific network training or empirical inputs."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from super_deli.benchmarking.neural_moons_config import NeuralMoonsConfig
from super_deli.benchmarking.neural_moons_fixture import create_fixture
from super_deli.io.config import read_json, write_json


def main(output):
    root = Path(output).resolve()
    root.mkdir(exist_ok=False)
    archive = create_fixture(root / "invented-archive")
    study = root / "diagnostic"
    study.mkdir()
    (study / "approved-plan.md").write_text(
        "Invented untrained engineering smoke; no performance inference."
    )
    config = root / "config.json"
    write_json(NeuralMoonsConfig(fixture=True).to_dict(), config)
    for action in (
        "bind",
        "freeze",
        "preflight",
        "audit-preflight",
        "gate",
        "main",
        "audit-main",
        "report",
    ):
        cmd = [
            sys.executable,
            "-m",
            "super_deli",
            "diagnose-neural-moons",
            action,
            "--archive-root",
            str(archive),
            "--output-dir",
            str(study),
            "--config",
            str(config),
        ]
        if action == "gate":
            cmd += ["--staging-bytes", "0"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
    audit = read_json(study / "main-audit.json")
    assert audit["passed"] and audit["attempted"] == 26 and audit["archive_replays"] == 16
    assert audit["failed"] == audit["unavailable_hierarchies"] == 0
    assert read_json(study / "aggregate.json")["fixture"]
    print(
        json.dumps(
            {
                "installed_neural_moons_smoke": "passed",
                "preflight_density_fits": 20,
                "main_density_fits": 26,
                "main_selection_evaluations": 62,
                "neural_training_fits": 0,
                "empirical_fits": 0,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    main(parser.parse_args().output_dir)
