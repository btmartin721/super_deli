"""New configuration remains isolated, strict and dependency-free."""

import json
import os
import subprocess
import sys

import pytest

from super_deli.models.denoising_config import DenoisingConfig


def test_roundtrip_and_invalid_contracts():
    c = DenoisingConfig(("a", "b", "c"), True)
    assert DenoisingConfig.from_dict(json.loads(json.dumps(c.to_dict()))) == c
    for change in (
        {"corruption_seed": True},
        {"corruption_seed": -1},
        {"corruption_sd": 0.3},
        {"model_family": "tabular_autoencoder_hdbscan"},
        {"schema_version": 2},
        {"assume_independent_units": False},
        {"feature_names": ["a", "b"]},
    ):
        with pytest.raises(ValueError):
            DenoisingConfig.from_dict({**c.to_dict(), **change})


def test_help_has_no_scientific_imports():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from super_deli.models.denoising_config import DenoisingConfig
from super_deli.cli import main
try:
    main(["cluster-tabular-denoising", "--help"])
except SystemExit as exc:
    assert exc.code == 0
assert not {"numpy", "scipy", "sklearn", "torch"}.intersection(sys.modules)
""",
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
