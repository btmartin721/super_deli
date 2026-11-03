"""Pure contracts, exact workload counts, separate streams and lazy commands."""

import subprocess
import sys
from dataclasses import replace

import pytest

from super_deli.benchmarking.matched_config import StudyConfig, counts, tasks
from super_deli.models.matched_neural_config import MatchedNeuralConfig


def config():
    return MatchedNeuralConfig("m", "e", ("m1", "m2", "m3"), ("e1", "e2", "e3"), True, "individual")


@pytest.mark.parametrize(
    "change",
    [
        {"assume_independent_units": False},
        {"assume_independent_units": 1},
        {"ecology_measurement_scope": "locality"},
        {"ecology": "m"},
        {"morphology": " m"},
        {"morphology_features": ("x",)},
        {"morphology_features": ("x", "x", "y")},
        {"ecology_features": (["x"], "y", "z")},
        {"ecology_features": tuple(str(i) for i in range(33))},
        {"schema_version": True},
        {"random_seed": True},
        {"random_seed": -1},
        {"n_subsamples": 2**32},
        {"max_working_bytes": 0},
        {"model_family": "tabular_autoencoder_hdbscan"},
    ],
)
def test_invalid_contract(change):
    with pytest.raises(ValueError):
        replace(config(), **change)


def test_roundtrip_and_unknown_fields():
    c = config()
    assert MatchedNeuralConfig.from_dict(c.to_dict()) == c
    for key in (
        "schema_version",
        "morphology",
        "ecology",
        "morphology_features",
        "ecology_features",
        "assume_independent_units",
        "ecology_measurement_scope",
    ):
        value = c.to_dict()
        value.pop(key)
        with pytest.raises(ValueError):
            MatchedNeuralConfig.from_dict(value)
    with pytest.raises(ValueError):
        MatchedNeuralConfig.from_dict({**c.to_dict(), "p_value": 0.05})


def test_frozen_counts_and_streams():
    c = StudyConfig()
    assert counts(c, "main") == {
        "observations": 160,
        "point_fits": 2240,
        "subsample_refits": 440,
        "point_replays": 28,
        "workflows": 2708,
        "neural_fits": 1818,
        "classical_fits": 890,
    }
    assert counts(c, "preflight")["workflows"] == 472
    assert counts(StudyConfig(fixture=True), "main")["workflows"] == 83
    assert counts(StudyConfig(fixture=True), "preflight")["workflows"] == 83
    pre, main = tasks(c, "preflight"), tasks(c, "main")
    for key in (
        "data_seed",
        "noise_seed",
        "permutation_seed",
        "ecology_order_seed",
        "subsampling_path",
    ):
        assert {tuple(t[key]) for t in pre}.isdisjoint(tuple(t[key]) for t in main)
    assert sum(t["draws"] > 0 for t in main) == 8 and sum(t["replay"] for t in main) == 2
    with pytest.raises(ValueError):
        StudyConfig(max_workers=3)
    with pytest.raises(ValueError):
        StudyConfig(max_projected_seconds=7201)


def test_help_is_lazy():
    code = """
import sys
from super_deli.cli import main
for command in ('cluster-matched-neural','benchmark-matched-neural'):
 try:main([command,'--help'])
 except SystemExit as e:assert e.code==0
assert not {'numpy','scipy','sklearn','torch'}.intersection(sys.modules)
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
