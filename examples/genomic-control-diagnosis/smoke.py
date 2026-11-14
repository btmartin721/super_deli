"""Installed-package fit-free audit on invented algebraic artifacts."""

import argparse
import json
from pathlib import Path

from synthetic import fixture

from super_deli.cli import main


def smoke(output):
    output.mkdir()
    config = fixture(output / "invented-archive")
    path = output / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    result = main(
        [
            "diagnose-genomic-neural",
            str(output / "invented-archive"),
            "--diagnosis-config",
            str(path),
            "--output-dir",
            str(output / "diagnosis"),
        ]
    )
    assert result == 0
    summary = json.loads((output / "diagnosis/summary.json").read_text())
    assert (
        summary["new_training_fits"]
        == summary["new_gmm_searches"]
        == summary["new_random_draws"]
        == 0
    )
    assert summary["point_models"] == summary["subsample_models"] == 5
    assert summary["decision"]["status"] == "invented_fixture_no_scientific_decision"
    print(json.dumps(dict(passed=True, archived_maps=10, new_fits=0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    smoke(parser.parse_args().output_dir)
