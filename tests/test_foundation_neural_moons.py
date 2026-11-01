"""Strict planning and optional-dependency isolation for the archival diagnostic."""

import os
import subprocess
import sys
from dataclasses import replace

import pytest

from super_deli.benchmarking.neural_moons_config import DENSITY_ARMS, NeuralMoonsConfig
from super_deli.benchmarking.neural_moons_study import plan


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("fixture", 1),
        ("max_workers", True),
        ("max_workers", 0),
        ("max_workers", 3),
        ("artifact_cap_bytes", -1),
        ("min_free_bytes", "0"),
        ("max_projected_seconds", 0),
        ("max_working_bytes", False),
    ],
)
def test_reject_ambiguous_settings(field, value):
    with pytest.raises(ValueError):
        replace(NeuralMoonsConfig(), **{field: value})


def test_contract_and_exact_fit_accounting():
    config = NeuralMoonsConfig()
    assert NeuralMoonsConfig.from_dict(config.to_dict()) == config
    for value in (None, [], {}, {"schema_version": 1, "choose_best": True}):
        with pytest.raises(ValueError):
            NeuralMoonsConfig.from_dict(value)
    binding = {
        "observations": [{"replicate": i} for _ in range(6) for i in range(10)],
        "replays": [{"replay_arm": a} for _ in range(2) for a in DENSITY_ARMS],
    }
    for phase, fits, selections in (("preflight", 56, 164), ("main", 548, 1628)):
        jobs = plan(binding, phase)
        assert sum(len(j["representations"]) for j in jobs) == fits
        assert (
            sum(len(j["representations"]) * (1 if j["replay_only"] else 3) for j in jobs)
            == selections
        )
    with pytest.raises(ValueError):
        plan(binding, "train")


def test_diagnostic_help_keeps_numerical_dependencies_optional():
    code = """
import sys
from super_deli.cli import main
from super_deli.benchmarking.neural_moons_config import NeuralMoonsConfig
try:
    main(["diagnose-neural-moons", "--help"])
except SystemExit as e:
    assert e.code == 0
assert not {"torch", "numpy", "scipy", "sklearn"} & sys.modules.keys()
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
