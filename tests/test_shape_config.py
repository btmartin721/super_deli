"""Shape policies and imports stay strict and dependency-free."""

import subprocess
import sys

import pytest

from super_deli.models.shape_config import ShapeConfig


def test_shape_config_roundtrip():
    config = ShapeConfig()
    assert ShapeConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"landmark_type": "sliding"},
        {"reflection": "automatic"},
        {"reflection": []},
        {"missing_policy": "mean"},
        {"repeated_policy": "average"},
        {"n_components": 0},
        {"n_components": False},
        {"n_components": 1.5},
        {"tol": 0},
        {"tol": True},
        {"tol": float("nan")},
        {"tol": float("inf")},
        {"rank_tol": 1},
        {"max_iter": 0},
        {"max_iter": 5.0},
        {"max_working_bytes": 0},
        {"unknown": 1},
    ],
)
def test_invalid_shape_config(change):
    with pytest.raises(ValueError):
        ShapeConfig.from_dict({"schema_version": 1, **change})


@pytest.mark.parametrize("document", [None, [], {}, {"n_components": 2}])
def test_required_config_schema(document):
    with pytest.raises(ValueError):
        ShapeConfig.from_dict(document)


def test_shape_workflow_imports_without_optional_libraries():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from super_deli.models.shape_config import ShapeConfig
from super_deli.workflows.shape_pca import run_shape_pca
from super_deli.io.shape_model import read_shape_model
assert not {'numpy','scipy','PIL','snpio','sklearn'} & sys.modules.keys()
""",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
