"""Installed CLI, native stability and numeric reload on invented data only."""

import argparse
from pathlib import Path

import numpy as np

from super_deli.benchmarking.neural_study import FEATURES, write_dataset
from super_deli.cli import main
from super_deli.io.config import read_json, write_json
from super_deli.models.denoising_config import DenoisingConfig
from super_deli.workflows.tabular_denoising import load_denoising_model


def smoke(directory):
    directory.mkdir(exist_ok=False)
    x = np.random.default_rng(785).normal(size=(32, 8))
    write_dataset(x, directory / "raw")
    write_json(
        DenoisingConfig(FEATURES, True, corruption_seed=786, n_subsamples=2).to_dict(),
        directory / "model.json",
    )
    assert (
        main(
            [
                "cluster-tabular-denoising",
                str(directory / "raw" / "dataset.json"),
                "--model-config",
                str(directory / "model.json"),
                "--output-dir",
                str(directory / "fit"),
            ]
        )
        == 0
    )
    model = load_denoising_model(directory / "fit")
    with np.load(directory / "fit" / "point.npz", allow_pickle=False) as f:
        np.testing.assert_allclose(model.encode(x), f["scores"], atol=1e-12)
    assert model.reconstruct(x).shape == x.shape
    summary = read_json(directory / "fit" / "summary.json")
    assert summary["stability"]["successful"] == 2
    assert summary["species_significance"]["status"] == "not_computed"
    print("Installed denoising CLI, native subsampling and numeric reload passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    smoke(parser.parse_args().output_dir)
