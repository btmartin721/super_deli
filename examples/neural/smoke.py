"""Installed neural CLI and frozen-model check on an invented development fixture."""

import argparse
from pathlib import Path

import numpy as np

from super_deli.benchmarking.neural_study import FEATURES, write_dataset
from super_deli.cli import main
from super_deli.io.config import read_json, write_json
from super_deli.models.neural_config import NeuralConfig
from super_deli.workflows.tabular_neural import load_neural_model


def smoke(directory):
    directory.mkdir(exist_ok=False)
    raw = np.random.default_rng(552).normal(size=(40, 8))
    write_dataset(raw, directory / "raw")
    write_json(NeuralConfig(FEATURES, True, n_subsamples=2).to_dict(), directory / "model.json")
    assert (
        main(
            [
                "cluster-tabular-neural",
                str(directory / "raw" / "dataset.json"),
                "--model-config",
                str(directory / "model.json"),
                "--output-dir",
                str(directory / "fit"),
            ]
        )
        == 0
    )
    model = load_neural_model(directory / "fit")
    with np.load(directory / "fit" / "point.npz", allow_pickle=False) as fitted:
        np.testing.assert_allclose(model.encode(raw), fitted["scores"], atol=1e-12)
    assert model.reconstruct(raw).shape == raw.shape
    assert read_json(directory / "fit" / "summary.json")["stability"]["successful"] == 2
    print("Installed neural CLI, subsampling and numeric-model roundtrip passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    smoke(parser.parse_args().output_dir)
