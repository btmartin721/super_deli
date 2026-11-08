"""Strict opt-in assumptions without importing scientific dependencies."""

from dataclasses import replace

import pytest

from super_deli.models.neural_calibration_config import NeuralCalibrationConfig
from super_deli.models.separate_gmm_config import SeparateGMMConfig


def config():
    return NeuralCalibrationConfig(
        True, SeparateGMMConfig("m", "e", ("a", "b", "c"), ("x", "y", "z"), True, "individual")
    )


def test_roundtrip():
    assert NeuralCalibrationConfig.from_dict(config().to_dict()) == config()


@pytest.mark.parametrize(
    "change",
    [
        {"assume_iid_gaussian_raw_views": False},
        {"n_simulations": True},
        {"n_simulations": 0},
        {"alpha": float("nan")},
        {"rank_tol": 0},
        {"random_seed": -1},
        {"schema_version": 2},
        {"max_archive_bytes": False},
    ],
)
def test_invalid_policy(change):
    with pytest.raises(ValueError):
        replace(config(), **change)


def test_sampling_and_single_component_search_rejected():
    for change in ({"n_subsamples": 1}, {"max_k": 1}):
        with pytest.raises(ValueError):
            replace(config(), model=replace(config().model, **change))


def test_unknown_and_missing_fields():
    for change in ({"unexpected": True}, {"model": {}}):
        with pytest.raises(ValueError):
            NeuralCalibrationConfig.from_dict({**config().to_dict(), **change})
