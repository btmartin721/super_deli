"""Invented matched measurements; not an empirical species-validation dataset."""

import argparse
from pathlib import Path

from super_deli.benchmarking.matched_data import configuration, generate, write_dataset
from super_deli.io.config import write_json


def generate_example(out):
    out.mkdir()
    task = {
        "case": "four_complementary",
        "n": 64,
        "data_seed": [109111],
        "noise_seed": [109117],
        "permutation_seed": [109121],
        "ecology_order_seed": [109123],
    }
    write_dataset(generate(task), out / "raw")
    write_json(configuration("paired_0", 109127).to_dict(), out / "model.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output_dir", type=Path)
    generate_example(p.parse_args().output_dir)
