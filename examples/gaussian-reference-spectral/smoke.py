"""Installed gated engineering fixture; no biological validation or neural training."""

import argparse
from pathlib import Path

from super_deli.benchmarking.gaussian_graph_cli import main
from super_deli.benchmarking.gaussian_graph_config import StudyConfig
from super_deli.io.config import read_json, write_json


def smoke(out):
    out.mkdir()
    root = out / "study"
    root.mkdir()
    (root / "approved-plan.md").write_text("Invented installed engineering fixture")
    config = out / "fixture.json"
    write_json(StudyConfig(fixture=True).to_dict(), config)
    for action in (
        "freeze",
        "preflight",
        "audit-preflight",
        "gate",
        "main",
        "audit-main",
        "report",
    ):
        args = [action, "--output-dir", str(root), "--study-config", str(config)]
        if action == "gate":
            args.extend(["--staging-bytes", "0"])
        assert main(args) == 0
    assert read_json(root / "main-audit.json")["actual_evaluation_attempts"] == 1800
    assert read_json(root / "aggregate.json")["fixture"]
    print("Installed benchmark-only Gaussian-reference fixture passed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    smoke(p.parse_args().output_dir)
