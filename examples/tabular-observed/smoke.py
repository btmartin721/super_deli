"""Installed opt-in workflow; all specimen identities are invented."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from super_deli import __version__
from super_deli.cli import main


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    assert __version__ == version("super-deli")
    assert (
        main(
            [
                "cluster-tabular-observed",
                str(base / "dataset.json"),
                "--model-config",
                str(base / "model.json"),
                "--output-dir",
                str(args.output_dir),
            ]
        )
        == 0
    )
    result = json.loads((args.output_dir / "summary.json").read_text())
    assert result["model"] == "observed_coordinate_full_gmm"
    assert result["p_value"] is None and result["excluded_all_missing"] == 1
    import numpy as np

    from super_deli.workflows.observed_gmm import load_observed_model

    model = load_observed_model(args.output_dir)
    density, membership = model.predict(np.array([[0.0, np.nan, 1.0, 2.0], [np.nan] * 4]))
    assert np.isfinite(density[0]) and np.isnan(density[1])
    assert np.isnan(membership[1]).all()
    print("Observed-data installed workflow passed")


if __name__ == "__main__":
    run()
