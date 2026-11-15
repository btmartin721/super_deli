"""Installed CLI/API smoke using only invented complete matched inputs."""

import argparse
from pathlib import Path

from synthetic import invented


def smoke(output):
    from super_deli.cli import main
    from super_deli.io.config import read_json

    output = Path(output)
    output.mkdir()
    invented(output / "source")
    status = main(
        [
            "analyze-matched-multimodal",
            str(output / "source/dataset.json"),
            "--analysis-config",
            str(output / "source/analysis.json"),
            "--output-dir",
            str(output / "analysis"),
        ]
    )
    summary = read_json(output / "analysis/summary.json")
    assert status == 0 and summary["status"] == "complete"
    assert len(summary["models"]) == 5 and len(summary["comparisons"]) == 10
    assert summary["inference"]["p_value"] is None
    assert all(row["stability"]["requested"] == 0 for row in summary["models"])
    print('{"passed": true, "models": 5, "n_units": 64}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    smoke(parser.parse_args().output_dir)
