"""The optional neural contract must remain dependency-free and strict."""

import json
import os
import subprocess
import sys

import pytest

from super_deli.models.neural_config import NeuralConfig


def test_roundtrip_and_unknown_fields():
    config = NeuralConfig(("a", "b", "c"), True)
    assert config == NeuralConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    for extra in (
        {"epochs": 1},
        {"schema_version": True},
        {"random_seed": -1},
        {"assume_independent_units": 1},
        {"n_subsamples": False},
        {"feature_names": ["a", "a", "b"]},
        {"feature_names": ["a", "b"]},
        {"feature_names": "abc"},
        {"model_family": "gmm"},
    ):
        with pytest.raises(ValueError):
            NeuralConfig.from_dict({**config.to_dict(), **extra})


def test_cli_and_configuration_do_not_import_neural_dependencies():
    code = """
import sys
from super_deli.models.neural_config import NeuralConfig
from super_deli.cli import main
assert not any(x in sys.modules for x in ("torch", "numpy", "scipy", "sklearn"))
try:
    main(["cluster-tabular-neural", "--help"])
except SystemExit as e:
    assert e.code == 0
assert "torch" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
