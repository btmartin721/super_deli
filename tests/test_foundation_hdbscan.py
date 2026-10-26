"""Dependency-free isolation, versioning, and unsupported-family contracts."""

from dataclasses import replace

import pytest

from super_deli.benchmarking.density_study import DensityStudyConfig
from super_deli.models.hdbscan_config import HDBSCANConfig
from super_deli.models.tabular_cluster_config import TabularClusterConfig


def test_roundtrip_and_no_gmm_configuration_reuse():
    c = HDBSCANConfig(True, feature_names=("x", "y"))
    assert HDBSCANConfig.from_dict(c.to_dict()) == c
    assert c.preprocessing_config().n_bootstrap == 0
    with pytest.raises(ValueError, match="unknown"):
        TabularClusterConfig.from_dict(c.to_dict())
    with pytest.raises(ValueError):
        HDBSCANConfig.from_dict(TabularClusterConfig(True).to_dict())


@pytest.mark.parametrize(
    "field,value",
    [
        ("assume_independent_units", False),
        ("assume_independent_units", 1),
        ("schema_version", True),
        ("schema_version", 2),
        ("model_family", "gmm"),
        ("n_pcs", 0),
        ("n_pcs", True),
        ("min_cluster_size", 1),
        ("min_samples", 0),
        ("alpha", 0),
        ("alpha", float("nan")),
        ("cluster_selection_epsilon", -1),
        ("cluster_selection_method", "invalid"),
        ("cluster_selection_method", []),
        ("allow_single_cluster", 1),
        ("n_subsamples", -1),
        ("n_subsamples", True),
        ("subsample_fraction", 0),
        ("subsample_fraction", 1),
        ("subsample_fraction", float("inf")),
        ("rank_tol", 0),
        ("rank_tol", 1),
        ("max_working_bytes", 0),
        ("max_pairwise_bytes", False),
        ("random_seed", -1),
        ("feature_names", ["x"]),
        ("feature_names", ("x", "x")),
        ("feature_names", (" x",)),
        ("feature_names", (1,)),
    ],
)
def test_reject_invalid_density_settings(field, value):
    with pytest.raises(ValueError):
        replace(HDBSCANConfig(True), **{field: value})


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"schema_version": 1},
        {
            "schema_version": 1,
            "model_family": "tabular_pca_hdbscan",
            "assume_independent_units": True,
            "max_k": 4,
        },
    ],
)
def test_require_versioned_specific_contract(value):
    with pytest.raises(ValueError):
        HDBSCANConfig.from_dict(value)


def test_frozen_budget_and_seed_streams():
    c = DensityStudyConfig()
    for phase, total in [("main", 3600), ("preflight", 624)]:
        rows = c.plan(phase)
        assert sum(2 * (1 + r["draws"]) for r in rows) == total
        assert len(set(tuple(r["data_seed"]) for r in rows)) == len(rows)
    assert {tuple(r["data_seed"]) for r in c.plan("main")}.isdisjoint(
        {tuple(r["data_seed"]) for r in c.plan("preflight")}
    )
    assert DensityStudyConfig.from_dict(c.to_dict()) == c
