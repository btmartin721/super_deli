"""The new configuration imports without scientific extras."""

import pytest

from super_deli.models.observed_config import ObservedGMMConfig


def test_round_trip_and_legacy_boundary():
    value = ObservedGMMConfig(("a", "b"))
    assert ObservedGMMConfig.from_dict(value.to_dict()) == value
    assert value.to_dict()["covariance_bound"] == 1e-4


@pytest.mark.parametrize(
    "changes",
    [
        dict(feature_names=["a", "b"]),
        dict(max_k=5),
        dict(n_init=True),
        dict(covariance_bound=0),
        dict(tol=float("nan")),
        dict(feature_names=("a", "a")),
        dict(schema_version=2),
        dict(min_coobserved=1),
    ],
)
def test_strict_contract(changes):
    with pytest.raises(ValueError):
        ObservedGMMConfig(**{**dict(feature_names=("a", "b")), **changes})


def test_unknown_keys():
    with pytest.raises(ValueError):
        ObservedGMMConfig.from_dict(dict(feature_names=["a", "b"], n_pcs=2))
