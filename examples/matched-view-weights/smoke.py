"""Installed schema-2 CLI and dependency-light frozen-transform check."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from super_deli.benchmarking.matched_data import write_dataset
from super_deli.benchmarking.view_weight_config import ViewWeightStudy, model_for, tasks
from super_deli.benchmarking.view_weight_data import generate
from super_deli.io.config import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    root = parser.parse_args().output_dir.resolve()
    root.mkdir(exist_ok=False)
    task = tasks(ViewWeightStudy(fixture=True), "main")[0]
    write_dataset(generate(task), root / "input")
    write_json(model_for("joint_morphology_0", 0).to_dict(), root / "model.json")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "super_deli",
            "cluster-matched-neural",
            str(root / "input/dataset.json"),
            "--model-config",
            str(root / "model.json"),
            "--output-dir",
            str(root / "fit"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    script = """
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch','snpio'}:
            raise RuntimeError('unexpected optional import: '+fullname)
sys.meta_path.insert(0,Block())
import numpy as np
from pathlib import Path
from super_deli.workflows.matched_neural import load_matched_neural_model
from super_deli.benchmarking.matched_audit import synthetic_tables
from super_deli.benchmarking.view_weight_config import ViewWeightStudy,tasks
from super_deli.benchmarking.view_weight_data import generate
root=Path(sys.argv[1]); model=load_matched_neural_model(root/'fit')
tables=synthetic_tables(generate(tasks(ViewWeightStudy(fixture=True),'main')[0]))
with np.load(root/'fit/point.npz',allow_pickle=False) as saved:
    np.testing.assert_array_equal(model.encode(tables)['scores'],saved['scores'])
assert model.reconstruct(tables)['views']['ecology'].shape==(32,8)
"""
    subprocess.run([sys.executable, "-c", script, str(root)], check=True)
    print(
        json.dumps({"passed": True, "weighted_cli": True, "frozen_transform_without_torch": True})
    )


if __name__ == "__main__":
    main()
