"""Pure schema-2 contracts without changing legacy serialization."""

from dataclasses import replace

import pytest

from super_deli.models.matched_neural_config import MatchedNeuralConfig


def legacy():
    return MatchedNeuralConfig("m", "e", ("a", "b", "c"), ("x", "y", "z"), True, "individual")


def test_legacy_schema_stays_exact():
    model = legacy()
    assert "view_loss_weights" not in model.to_dict()
    assert MatchedNeuralConfig.from_dict(model.to_dict()) == model
    with pytest.raises(ValueError, match="Schema-1"):
        MatchedNeuralConfig.from_dict({**model.to_dict(), "view_loss_weights": None})
    with pytest.raises(ValueError, match="schema 2"):
        replace(model, view_loss_weights=(0.8, 0.2))


@pytest.mark.parametrize("weights", [(0.5, 0.5), (0.8, 0.2), (0.2, 0.8)])
def test_explicit_fractions_roundtrip(weights):
    c = replace(legacy(), schema_version=2, view_loss_weights=weights)
    value = c.to_dict()
    value["view_loss_weights"] = list(weights)
    assert MatchedNeuralConfig.from_dict(value) == c
    assert c.loss_fractions() == weights


@pytest.mark.parametrize(
    "weights",
    [
        None,
        (),
        (1.0,),
        (0.0, 1.0),
        (1.0, 0.0),
        (-0.1, 1.1),
        (0.5, 0.4),
        (float("nan"), 0.5),
        (float("inf"), 0.5),
        (True, 0.2),
        (0.3, "0.7"),
        [0.5, 0.5],
    ],
)
def test_invalid_fractions_rejected(weights):
    with pytest.raises(ValueError):
        replace(legacy(), schema_version=2, view_loss_weights=weights)


def test_schema_two_requires_weights():
    with pytest.raises(ValueError):
        MatchedNeuralConfig.from_dict({**legacy().to_dict(), "schema_version": 2})
