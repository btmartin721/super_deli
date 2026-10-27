"""Versioned opt-in configuration without numerical imports."""

from dataclasses import FrozenInstanceError

import pytest

from super_deli.models.hdbscan_config import HDBSCANConfig
from super_deli.models.tabular_analysis_config import TabularAnalysisArm, TabularAnalysisConfig
from super_deli.models.tabular_cluster_config import TabularClusterConfig
from super_deli.models.tabular_missing_config import TabularMissingAnalysisConfig


def config():
    names = ("a", "b")
    return TabularMissingAnalysisConfig(
        (
            TabularAnalysisArm(
                "gmm", TabularClusterConfig(True, feature_names=names, n_bootstrap=0)
            ),
            TabularAnalysisArm("density", HDBSCANConfig(True, feature_names=names)),
        ),
        ("complete_case", "mean"),
    )


def test_separate_schema_roundtrip_does_not_change_legacy():
    c = config()
    assert TabularAnalysisConfig.from_dict(c.to_dict()) == c
    old = TabularAnalysisConfig(c.arms)
    assert TabularAnalysisConfig.from_dict(old.to_dict()).to_dict() == old.to_dict()
    assert set(old.to_dict()) == {"arms", "schema_version"}
    assert "missing_policy" not in c.arms[1].model.to_dict()
    with pytest.raises(FrozenInstanceError):
        c.schema_version = 1


@pytest.mark.parametrize(
    "policies", [(), ("bad",), ("mean", "mean"), ("mean", "complete_case", "mean")]
)
def test_invalid_policies(policies):
    with pytest.raises(ValueError):
        TabularMissingAnalysisConfig(config().arms, policies)


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", True),
        ("schema_version", 1),
        ("policies", "mean"),
        ("arms", {}),
        ("extra", True),
    ],
)
def test_bad_contract(key, value):
    v = config().to_dict()
    v[key] = value
    with pytest.raises(ValueError):
        TabularMissingAnalysisConfig.from_dict(v)
