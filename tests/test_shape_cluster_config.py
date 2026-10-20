"""Strict assumptions and optional imports for the shape mixture checkpoint."""

import subprocess
import sys

import pytest

from super_deli.models.shape_cluster_config import ShapeClusterConfig


def test_roundtrip_and_explicit_shape_dimension():
    config = ShapeClusterConfig(True)
    assert ShapeClusterConfig.from_dict(config.to_dict()) == config
    assert config.shape_config.n_components == config.n_pcs


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"unknown": 1},
        {"assume_independent_units": False},
        {"assume_independent_units": 1},
        {"n_pcs": 0},
        {"n_pcs": True},
        {"n_bootstrap": -1},
        {"n_bootstrap": 1.5},
        {"min_cluster_size": 2},
        {"max_k": 0},
        {"n_init": 0},
        {"max_iter": 0},
        {"random_seed": -1},
        {"n_threads": 0},
        {"max_pairwise_bytes": 0},
        {"tol": True},
        {"tol": float("nan")},
        {"tol": 0},
        {"reg_covar": 0},
        {"reg_covar": float("inf")},
        {"assignment_threshold": -0.1},
        {"assignment_threshold": 1.1},
        {"shape": None},
        {"shape": {"schema_version": 1, "repeated_policy": "allow"}},
        {"shape": {"schema_version": 1, "n_components": 3}},
    ],
)
def test_invalid_config(change):
    with pytest.raises(ValueError):
        ShapeClusterConfig.from_dict(
            {"schema_version": 1, "assume_independent_units": True, **change}
        )


@pytest.mark.parametrize("value", [None, [], {}, {"schema_version": 1}])
def test_required_fields(value):
    with pytest.raises(ValueError):
        ShapeClusterConfig.from_dict(value)


def test_imports_are_dependency_free():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from super_deli.models.shape_cluster_config import ShapeClusterConfig
from super_deli.workflows.shape_gmm import run_shape_gmm
assert not {'numpy', 'scipy', 'sklearn', 'snpio', 'PIL'} & sys.modules.keys()
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
