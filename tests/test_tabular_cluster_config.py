"""Dependency-free, strict configuration for numeric morphology/ecology baselines."""

import subprocess
import sys

import pytest

from super_deli.models.tabular_cluster_config import TabularClusterConfig


def test_roundtrip_and_feature_order():
    config = TabularClusterConfig(True, feature_names=("length", "mass"))
    assert TabularClusterConfig.from_dict(config.to_dict()) == config
    assert TabularClusterConfig.from_dict(
        {
            "schema_version": 1,
            "assume_independent_units": True,
            "feature_names": ["mass", "length"],
        }
    ).feature_names == ("mass", "length")


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
        {"min_cluster_size": 2},
        {"n_bootstrap": -1},
        {"random_seed": True},
        {"max_k": 0},
        {"n_init": 0},
        {"max_iter": 0},
        {"n_threads": 0},
        {"max_pairwise_bytes": 0},
        {"tol": float("nan")},
        {"reg_covar": 0},
        {"assignment_threshold": 1.1},
        {"feature_names": "length"},
        {"feature_names": ["length", "length"]},
        {"feature_names": [" "]},
        {"feature_names": [1]},
        {"missing_policy": "auto"},
        {"scaling": []},
        {"count_transform": "sqrt"},
        {"rank_tol": 0},
        {"rank_tol": True},
        {"rank_tol": float("inf")},
        {"max_working_bytes": 0},
        {"max_working_bytes": 2.5},
    ],
)
def test_invalid_config(change):
    with pytest.raises(ValueError):
        TabularClusterConfig.from_dict(
            {"schema_version": 1, "assume_independent_units": True, **change}
        )


@pytest.mark.parametrize("value", [None, [], {}, {"schema_version": 1}])
def test_required_config(value):
    with pytest.raises(ValueError):
        TabularClusterConfig.from_dict(value)


def test_imports_do_not_load_scientific_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from super_deli.models.tabular_cluster_config import TabularClusterConfig
from super_deli.workflows.tabular_gmm import run_tabular_gmm
assert not {'numpy', 'scipy', 'sklearn', 'snpio', 'PIL'} & sys.modules.keys()
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
