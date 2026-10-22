"""Dependency-free validation of explicit calibration assumptions and bounds."""

import pytest

from super_deli.models.calibration_config import GaussianCalibrationConfig


def test_roundtrip_and_explicit_assumption():
    config = GaussianCalibrationConfig(True)
    assert GaussianCalibrationConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError):
        GaussianCalibrationConfig.from_dict({"schema_version": 1})


@pytest.mark.parametrize(
    "options",
    [
        {"assume_iid_gaussian_scores": False},
        {"assume_iid_gaussian_scores": 1},
        {"n_simulations": 0},
        {"n_simulations": True},
        {"alpha": 0},
        {"alpha": 1},
        {"alpha": float("nan")},
        {"random_seed": -1},
        {"random_seed": False},
        {"rank_tol": 0},
        {"max_working_bytes": 0},
        {"schema_version": 2},
    ],
)
def test_invalid_policy(options):
    with pytest.raises(ValueError):
        GaussianCalibrationConfig(**{"assume_iid_gaussian_scores": True, **options})


def test_unknown_options_rejected():
    with pytest.raises(ValueError):
        GaussianCalibrationConfig.from_dict(
            {
                "schema_version": 1,
                "assume_iid_gaussian_scores": True,
                "species_test": True,
            }
        )
