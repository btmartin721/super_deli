"""Pure separate-encoder schema and optional-dependency boundaries."""

import subprocess
import sys
from dataclasses import replace

import pytest

from super_deli.models.separate_config import SeparateNeuralConfig


def config():
    return SeparateNeuralConfig("m", "e", ("a", "b", "c"), ("x", "y", "z"), True, "individual")


def test_config_roundtrip():
    c = config()
    assert SeparateNeuralConfig.from_dict(c.to_dict()) == c
    assert c.n_subsamples == 0 and "view_loss_weights" not in c.to_dict()
    with pytest.raises(ValueError):
        SeparateNeuralConfig.from_dict({**c.to_dict(), "view_loss_weights": [0.5, 0.5]})


@pytest.mark.parametrize(
    "options",
    [
        {"schema_version": 2},
        {"model_family": "other"},
        {"assume_independent_units": False},
        {"ecology_measurement_scope": "locality"},
        {"morphology_features": ("a", "b")},
        {"ecology": "m"},
        {"random_seed": True},
        {"max_working_bytes": 0},
    ],
)
def test_invalid_config(options):
    with pytest.raises(ValueError):
        replace(config(), **options)


def test_lazy_cli_help():
    script = """
import sys
class Block:
    def find_spec(self,fullname,*args):
        if fullname.split('.')[0] in {'numpy','scipy','sklearn','torch','snpio'}:
            raise RuntimeError('unexpected optional import')
sys.meta_path.insert(0,Block())
from super_deli.cli import main
main(['cluster-matched-separate-neural','--help'])
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
