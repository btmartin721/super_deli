"""Dependency-free explicit joint scientific-boundary declarations."""

from dataclasses import replace
from pathlib import Path

import pytest

from super_deli.io.config import read_json
from super_deli.models.joint_config import JointCalibrationConfig, JointStudyConfig

ROOT = Path(__file__).resolve().parents[1]


def policy():
    return JointCalibrationConfig.from_dict(
        read_json(ROOT / "examples/joint-calibration/calibration.json")
    )


def test_roundtrip_and_optional_single_continuous_view():
    c = policy()
    assert JointCalibrationConfig.from_dict(c.to_dict()) == c
    s = JointStudyConfig.from_dict(read_json(ROOT / "examples/joint-calibration/study.json"))
    assert JointStudyConfig.from_dict(s.to_dict()) == s
    for i in (1, 2):
        m = replace(c.model, views=(c.model.views[0], c.model.views[i]))
        assert replace(c, model=m).model.views == m.views


@pytest.mark.parametrize(
    "name",
    [
        "assume_unrelated_specimens",
        "assume_independent_blocks",
        "assume_fixed_panel",
        "assume_prespecified_blocks",
        "assume_conditional_gaussian",
        "assume_prespecified_predictors",
    ],
)
def test_explicit_assumptions(name):
    with pytest.raises(ValueError):
        replace(policy(), **{name: False})


@pytest.mark.parametrize(
    "updates",
    [
        {"predictor_blocks": ()},
        {"predictor_blocks": ("absent",)},
        {"predictor_blocks": ("block_00", "block_00")},
        {"predictor_blocks": ["block_00"]},
        {"block_ids": ("one",) * 128},
        {"rank_tol": float("nan")},
        {"n_simulations": True},
        {"alpha": 0},
        {"random_seed": -1},
        {"schema_version": 2},
    ],
)
def test_invalid_policies(updates):
    with pytest.raises(ValueError):
        replace(policy(), **updates)


def test_reject_missing_grouped_and_stability():
    c = policy()
    for kwargs in (
        {"n_bootstrap": 1},
        {"target_level": "locality"},
        {"missing_views": "complete_cases"},
    ):
        with pytest.raises(ValueError):
            replace(c, model=replace(c.model, **kwargs))
