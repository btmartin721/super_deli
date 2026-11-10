"""Installed-package invented lifecycle; no empirical or scientific-study data."""

import argparse
import json
from pathlib import Path

from super_deli.benchmarking.neural_power_archive import bind
from super_deli.benchmarking.neural_power_budget import gate
from super_deli.benchmarking.neural_power_config import NeuralPowerStudy
from super_deli.benchmarking.neural_power_fixture import create
from super_deli.benchmarking.neural_power_report import report
from super_deli.benchmarking.neural_power_study import engine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    base = args.output_dir
    base.mkdir()
    source, root = base / "source", base / "study"
    create(source)
    root.mkdir()
    (root / "approved-plan.md").write_text("Invented installation fixture")
    bind(source, root, fixture=True)
    study = engine(root, source)
    study.freeze(NeuralPowerStudy(fixture=True))
    study.run("preflight", workers=2)
    study.audit("preflight")
    assert gate(root, source, staging_bytes=0)["passed"]
    study.run("main", workers=2)
    study.audit("main")
    value = report(root, source)
    assert value["audits"]["main"]["verified_optimizer_updates"] == 6000
    print(json.dumps(dict(passed=True, workflows=value["counts"]["main"]["workflows"])))


if __name__ == "__main__":
    main()
