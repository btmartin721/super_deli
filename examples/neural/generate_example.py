"""Create an invented eight-feature input; no empirical measurements or species truth."""

import argparse
from pathlib import Path

import numpy as np

from super_deli.benchmarking.neural_study import FEATURES, write_dataset
from super_deli.io.config import write_json
from super_deli.models.neural_config import NeuralConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="New directory for an invented example")
    target = parser.parse_args().output_dir
    target.mkdir(exist_ok=False)
    raw = np.random.default_rng(552).normal(size=(40, 8))
    write_dataset(raw, target / "raw")
    write_json(NeuralConfig(FEATURES, True, n_subsamples=2).to_dict(), target / "model.json")
