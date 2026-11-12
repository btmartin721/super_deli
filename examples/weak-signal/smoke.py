"""Installed, invented diagnostic lifecycle; no neural training or empirical data."""

import argparse
import json
from pathlib import Path

from super_deli.benchmarking.coordinate_fit import frozen_only
from super_deli.benchmarking.weak_signal_config import WeakSignalStudy
from super_deli.benchmarking.weak_signal_fixture import create
from super_deli.benchmarking.weak_signal_report import report
from super_deli.benchmarking.weak_signal_source import bind
from super_deli.benchmarking.weak_signal_study import engine, gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    base = parser.parse_args().output_dir
    base.mkdir()
    source, root = base / "source", base / "study"
    with frozen_only():
        create(source)
        root.mkdir()
        (root / "approved-plan.md").write_text("Invented engineering installation protocol")
        bind(source, root, fixture=True)
        study = engine(root, source)
        study.freeze(WeakSignalStudy(fixture=True))
        study.run("preflight", workers=2)
        study.audit("preflight")
        assert gate(root, source, staging_bytes=0)["passed"]
        study.run("main", workers=2)
        study.audit("main")
        result = report(root, source)
        assert result["observations"] == 8 and result["component_fits"] == 28
        assert result["reference_evaluations"] == 24 and result["neural_training"] == 0
    print(json.dumps(dict(passed=True, component_fits=28, neural_training=0)))


if __name__ == "__main__":
    main()
