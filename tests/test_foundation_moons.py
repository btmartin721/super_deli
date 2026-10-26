"""Dependency-free fixed diagnostic configuration and lazy CLI checks."""

from dataclasses import replace

import pytest

from super_deli.benchmarking.moons_config import ARMS, MoonsDiagnosticConfig, arm_settings


def test_frozen_settings_and_streams():
    c = MoonsDiagnosticConfig()
    assert len(ARMS) == 8
    assert len(c.fresh_plan("main")) == 150
    assert len(c.fresh_plan("preflight")) == 6
    assert set(tuple(r["seed"]) for r in c.fresh_plan("main")).isdisjoint(
        tuple(r["seed"]) for r in c.fresh_plan("preflight")
    )
    assert MoonsDiagnosticConfig.from_dict(c.to_dict()) == c
    assert arm_settings("baseline")[0].allow_single_cluster
    assert arm_settings("center_only")[1] == "center"
    with pytest.raises(ValueError):
        arm_settings("best")


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("datasets_per_cell", 0),
        ("datasets_per_cell", True),
        ("sample_sizes", [64]),
        ("sample_sizes", (64, 64)),
        ("sample_sizes", (31,)),
        ("sample_sizes", ()),
        ("artifact_cap_bytes", 0),
        ("min_free_bytes", 0),
        ("main_seed", 129737),
    ],
)
def test_reject_invalid_settings(field, value):
    with pytest.raises(ValueError):
        replace(MoonsDiagnosticConfig(), **{field: value})


@pytest.mark.parametrize("value", [None, [], {}, {"schema_version": 1, "choose_best": True}])
def test_reject_ambiguous_contract(value):
    with pytest.raises(ValueError):
        MoonsDiagnosticConfig.from_dict(value)
