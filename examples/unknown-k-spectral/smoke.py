"""Installed engineering fixtures; never empirical species validation."""

import argparse
from pathlib import Path

import numpy as np

from super_deli.benchmarking.neural_study import write_dataset
from super_deli.benchmarking.unknown_k_config import StudyConfig
from super_deli.benchmarking.unknown_k_data import model_config
from super_deli.cli import main
from super_deli.io.config import read_json, write_json


def smoke(out):
    out.mkdir()
    write_dataset(np.random.default_rng(99101).normal(size=(40, 8)), out / "raw")
    write_json(model_config("spectral_15").to_dict(), out / "model.json")
    assert (
        main(
            [
                "cluster-tabular-spectral",
                str(out / "raw/dataset.json"),
                "--model-config",
                str(out / "model.json"),
                "--output-dir",
                str(out / "point"),
            ]
        )
        == 0
    )
    summary = read_json(out / "point/summary.json")
    assert summary["status"] in {"available", "withheld"}
    assert summary["inference"]["p_value"] is None
    assert summary["stability"]["status"] == "not_computed"
    assert (out / "point/assignments.csv").exists() == (summary["status"] == "available")
    root = out / "study"
    root.mkdir()
    (root / "approved-plan.md").write_text("Invented engineering fixture, not a biological study")
    write_json(StudyConfig(fixture=True).to_dict(), out / "study-config.json")
    for action in (
        "freeze",
        "preflight",
        "audit-preflight",
        "gate",
        "main",
        "audit-main",
        "report",
    ):
        args = [
            "benchmark-unknown-k-spectral",
            action,
            "--output-dir",
            str(root),
            "--study-config",
            str(out / "study-config.json"),
        ]
        if action == "gate":
            args.extend(["--staging-bytes", "0"])
        assert main(args) == 0, action
    assert read_json(root / "preflight-audit.json")["evaluations"] == 74
    assert read_json(root / "main-audit.json")["evaluations"] == 64
    assert read_json(root / "aggregate.json")["fixture"]
    print("Installed unknown-K point model and gated fixture passed; zero neural training")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    smoke(p.parse_args().output_dir)
