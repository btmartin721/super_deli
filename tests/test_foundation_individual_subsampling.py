"""Opt-in configuration contracts remain dependency-free and preserve legacy bytes."""

import json
from dataclasses import replace

import pytest

from super_deli.models.individual_subsample_config import (
    IndividualSubsampleConfig,
    require_supported_point,
)
from super_deli.models.integrative_config import IntegrativeConfig, ViewConfig


def test_explicit_sampling_declaration():
    c = IndividualSubsampleConfig(25, 1601)
    assert IndividualSubsampleConfig.from_dict(c.to_dict()) == c
    for kw in (
        {"n_subsamples": 0},
        {"n_subsamples": 1001},
        {"random_seed": True},
        {"random_seed": -1},
        {"fraction": 1.0},
        {"fraction": 0.75},
        {"schema_version": True},
        {"policy": "bootstrap"},
    ):
        with pytest.raises(ValueError):
            replace(c, **kw)
    with pytest.raises(ValueError):
        IndividualSubsampleConfig.from_dict({**c.to_dict(), "unknown": 1})
    with pytest.raises(ValueError):
        IndividualSubsampleConfig.from_dict({"n_subsamples": 25})


def test_model_boundary_and_legacy_serialization():
    m = IntegrativeConfig(
        True,
        views=(ViewConfig("traits", "morphological"), ViewConfig("eco", "ecological")),
        n_pcs=4,
        min_cluster_size=5,
        n_init=100,
        covariance_type="tied",
        fusion_policy="retain_all_v1",
    )
    before = json.dumps(m.to_dict(), sort_keys=True)
    require_supported_point(m)
    assert before == json.dumps(m.to_dict(), sort_keys=True)
    for kw in (
        {"n_init": 10},
        {"initialization_policy": "diverse_v1"},
        {"n_threads": 2},
        {"fusion_policy": "joint_pca"},
    ):
        with pytest.raises(ValueError):
            require_supported_point(replace(m, **kw))


def test_cli_is_lazy(capsys):
    from super_deli.cli import main

    with pytest.raises(SystemExit) as e:
        main(["subsample-integrative", "--help"])
    assert e.value.code == 0 and "--stability-config" in capsys.readouterr().out
