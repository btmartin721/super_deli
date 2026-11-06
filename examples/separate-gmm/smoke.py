"""Installed matched/shared-GMM fit plus a fresh Torch-blocked numeric reload."""

import argparse
import subprocess
import sys
from pathlib import Path

from super_deli.benchmarking.separate_gmm_fixture import create_fixture
from super_deli.io.config import load_config, read_json
from super_deli.models.separate_gmm_config import FAMILY, SeparateGMMConfig
from super_deli.workflows.separate_gmm import run_separate_gmm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True, type=Path)
    root = p.parse_args().output_dir.resolve()
    root.mkdir()
    create_fixture(root / "source")
    dataset = load_config(root / "source/legacy/raw/dataset.json")
    config = SeparateGMMConfig.from_dict(
        {
            **read_json(root / "source/legacy/separate_0/model-config.json"),
            "model_family": FAMILY,
            "n_subsamples": 1,
        }
    )
    result = run_separate_gmm(
        dataset, model_config=config, base_dir=root / "source/legacy/raw", output_dir=root / "fit"
    )
    assert (
        result["covariance_type"] == "tied"
        and result["species_significance"]["status"] == "not_computed"
    )
    code = """
import importlib.abc,sys,numpy as np
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'torch','snpio'}:
   raise RuntimeError('Forbidden inference import '+name)
sys.meta_path.insert(0,Block())
from super_deli.workflows.separate_gmm import load_separate_gmm_model
from super_deli.io.config import load_config
from super_deli.io.tabular import ingest_tabular
r=Path(sys.argv[1]);ds=load_config(r/'source/legacy/raw/dataset.json')
tables=ingest_tabular(ds,base_dir=r/'source/legacy/raw',output_dir=r/'inference-ingestion').tables
model=load_separate_gmm_model(r/'fit');prediction=model.predict(tables)
np.testing.assert_allclose(prediction['membership'],model.arrays['membership'],atol=1e-8)
assert 'torch' not in sys.modules and 'snpio' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(root)], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    print("Installed shared-GMM fit, whole-ID refit and Torch-free numeric inference passed")


if __name__ == "__main__":
    main()
