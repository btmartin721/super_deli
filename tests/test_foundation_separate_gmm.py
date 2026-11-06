"""Pure schema boundaries and optional-import independence."""

import json
import subprocess
import sys

import pytest

from super_deli.models.separate_gmm_config import FAMILY, SeparateGMMConfig


def specification():
    return dict(
        schema_version=1,
        model_family=FAMILY,
        morphology="traits",
        ecology="environment",
        morphology_features=["a", "b", "c", "d"],
        ecology_features=["e", "f", "g"],
        assume_independent_units=True,
        ecology_measurement_scope="individual",
    )


def test_new_schema_preserves_the_legacy_encoder_contract():
    c = SeparateGMMConfig.from_dict(specification())
    assert SeparateGMMConfig.from_dict(json.loads(json.dumps(c.to_dict()))) == c
    assert c.encoder_config().model_family == "matched_separate_autoencoders_hdbscan"
    assert c.encoder_config().random_seed == c.random_seed
    assert c.mixture_seed == 71323


@pytest.mark.parametrize(
    "change",
    [
        dict(model_family="wrong"),
        dict(schema_version=2),
        dict(assume_independent_units=False),
        dict(ecology_measurement_scope="locality"),
        dict(max_k=0),
        dict(n_init=True),
        dict(min_cluster_size=4),
        dict(tol=float("nan")),
        dict(reg_covar=0),
        dict(mixture_seed=-1),
        dict(covariance_type="full"),
    ],
)
def test_invalid_or_unvalidated_contracts_are_rejected(change):
    with pytest.raises(ValueError):
        SeparateGMMConfig.from_dict({**specification(), **change})


def test_pure_cli_and_configuration_do_not_import_optional_libraries():
    code = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'numpy','scipy','sklearn','torch','snpio'}:
   raise RuntimeError(fullname)
sys.meta_path.insert(0,Block())
from super_deli.models.separate_gmm_config import SeparateGMMConfig
from super_deli.cli import main
try: main(['cluster-matched-separate-gmm','--help'])
except SystemExit as exc: assert exc.code == 0
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
