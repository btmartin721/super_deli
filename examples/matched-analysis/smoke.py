"""Installed-package CLI/API smoke on invented matched measurements only."""

import argparse
import json
from pathlib import Path

from super_deli.cli import main


def smoke(output):
    source = Path(__file__).resolve().parent
    result = main(
        [
            "analyze-matched-tabular",
            str(source / "dataset.json"),
            "--analysis-config",
            str(source / "analysis.json"),
            "--output-dir",
            str(output),
        ]
    )
    if result:
        raise RuntimeError("Matched comparison failed")
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "complete" and summary["inference"]["status"] == "not_computed"
    assert len(summary["models"]) == 2 and len(summary["comparisons"]) == 1
    assert all(r["stability_status"] == "not_computed" for r in summary["models"])
    report = (output / "report.html").read_text()
    assert "Morphology latent 1 / 2" in report and "Joint PC1 / PC2" in report
    assert "Absolute BIC values are not compared" in report
    print(json.dumps(dict(passed=True, n_units=summary["n_units"], models=2)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    smoke(p.parse_args().output_dir)
