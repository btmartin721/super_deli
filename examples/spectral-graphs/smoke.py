"""Installed supplied-K protocol on explicitly untrained invented coordinates."""

import argparse
from pathlib import Path

from super_deli.benchmarking.spectral_config import SpectralConfig
from super_deli.benchmarking.spectral_fixture import create_fixture
from super_deli.cli import main
from super_deli.io.config import read_json, write_json


def smoke(out):
    out.mkdir()
    source = create_fixture(out / "source")
    root = out / "diagnostic"
    root.mkdir()
    (root / "approved-plan.md").write_text(
        "Invented untrained installed fixture; no species claim.\n"
    )
    write_json(SpectralConfig(fixture=True).to_dict(), out / "config.json")
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
        args = [
            "diagnose-spectral-graphs",
            action,
            "--archive-root",
            str(source),
            "--output-dir",
            str(root),
            "--config",
            str(out / "config.json"),
        ]
        if action == "gate":
            args.extend(["--staging-bytes", "0"])
        assert main(args) == 0, action
    report = read_json(root / "aggregate.json")
    assert report["fixture"] and report["selected_k"] is None
    assert read_json(root / "main-audit.json")["evaluations"] == 848
    print("Installed spectral graph protocol passed; zero neural training; supplied K only")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    smoke(p.parse_args().output_dir)
