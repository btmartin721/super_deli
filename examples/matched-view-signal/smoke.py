"""Installed-package engineering check; no neural training or scientific data."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from super_deli.benchmarking.view_signal_fixture import create_fixture
from super_deli.io.config import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(exist_ok=False)
    source = create_fixture(root / "invented-archive")
    out = root / "diagnostic"
    out.mkdir()
    (out / "approved-plan.md").write_text("Invented installed-package check; no scientific fits.\n")
    config = root / "configuration.json"
    write_json({"schema_version": 1, "fixture": True}, config)
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
        command = [
            sys.executable,
            "-m",
            "super_deli",
            "diagnose-matched-view-signal",
            action,
            "--archive-root",
            str(source),
            "--output-dir",
            str(out),
            "--config",
            str(config),
        ]
        if action == "gate":
            command += ["--staging-bytes", "0"]
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        summary = json.loads(result.stdout)
        assert summary.get("passed", True), summary
    final = read_json(out / "summary.json")
    assert final["main_fits"] == 21 and final["main_audit"]["native_replays"] == 11
    assert final["new_neural_training"] == 0
    print(json.dumps({"passed": True, "kind": "invented_untrained_fixture", "main_fits": 21}))


if __name__ == "__main__":
    main()
