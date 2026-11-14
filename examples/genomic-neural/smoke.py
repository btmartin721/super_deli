"""Installed CLI, complete-genotype model roundtrip and PCA parity smoke."""

import argparse
import json
from pathlib import Path

import numpy as np

from super_deli.cli import main
from super_deli.models.genomic_neural import load_genomic_model


def smoke(output):
    source = Path(__file__).resolve().parent
    code = main(
        [
            "analyze-genomic-models",
            str(source / "dataset.json"),
            "--model-config",
            str(source / "analysis.json"),
            "--output-dir",
            str(output),
        ]
    )
    assert code == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "complete" and len(summary["models"]) == 5
    assert summary["inference"]["p_value"] is None
    assert len(summary["comparisons"]) == 10
    raw = np.load(output / "inputs/dosages.npy", allow_pickle=False)
    for row in summary["models"]:
        model = load_genomic_model(output / "arms" / row["name"])
        result = model.predict(raw, locus_keys=model.locus_keys)
        np.testing.assert_allclose(
            result["membership"], model.arrays["membership"], rtol=1e-10, atol=1e-12
        )
        assert row["stability"]["status"] == "not_computed"
    assert (output / "report.html").is_file()
    print(json.dumps(dict(passed=True, models=5, n_units=len(raw))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    smoke(parser.parse_args().output_dir)
