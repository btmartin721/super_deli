"""Versioned configuration and lazy CLI contracts without scientific imports."""

import subprocess
import sys
from dataclasses import replace

import pytest

from super_deli.benchmarking.unknown_k_config import StudyConfig, counts, tasks
from super_deli.models.spectral_config import SpectralConfig


@pytest.mark.parametrize(
    "change",
    [
        {"assume_independent_units": False},
        {"assume_independent_units": 1},
        {"feature_names": ("a",)},
        {"feature_names": ("a", "a")},
        {"feature_names": (" a", "b")},
        {"n_neighbors": True},
        {"n_neighbors": 10},
        {"max_k": 1},
        {"max_k": 2.5},
        {"min_cluster_size": 1},
        {"random_seed": -1},
        {"max_working_bytes": True},
        {"schema_version": True},
        {"model_family": "gmm"},
    ],
)
def test_invalid_config(change):
    with pytest.raises(ValueError):
        replace(SpectralConfig(True, ("a", "b"), 15), **change)


def test_config_roundtrip_explicit_neighbors_and_scope():
    c = SpectralConfig(True, ("a", "b"), 5)
    assert SpectralConfig.from_dict(c.to_dict()) == c
    for key in ("n_neighbors", "feature_names", "assume_independent_units", "schema_version"):
        data = c.to_dict()
        data.pop(key)
        with pytest.raises(ValueError):
            SpectralConfig.from_dict(data)
    with pytest.raises(ValueError):
        SpectralConfig.from_dict({**c.to_dict(), "n_bootstrap": 100})
    assert counts(StudyConfig(), "main") == {
        "observations": 320,
        "workflows": 2592,
        "probes": 0,
        "evaluations": 2592,
    }
    assert counts(StudyConfig(), "preflight")["evaluations"] == 154
    assert counts(StudyConfig(fixture=True), "main")["evaluations"] == 64
    assert counts(StudyConfig(fixture=True), "preflight")["evaluations"] == 74
    assert {tuple(t["data_seed"]) for t in tasks(StudyConfig(), "main")}.isdisjoint(
        tuple(t["data_seed"]) for t in tasks(StudyConfig(), "preflight")
    )


@pytest.mark.parametrize(
    "change",
    [
        {"fixture": 1},
        {"max_workers": 3},
        {"artifact_cap_bytes": 2**32},
        {"min_free_bytes": 1},
        {"max_projected_seconds": 3601},
    ],
)
def test_study_limits(change):
    with pytest.raises(ValueError):
        StudyConfig(**change)


def test_help_is_lazy():
    code = """
import sys
from super_deli.cli import main
for command in ("cluster-tabular-spectral","benchmark-unknown-k-spectral"):
    try: main([command,"--help"])
    except SystemExit as exc: assert exc.code == 0
assert not {"numpy","scipy","sklearn","torch"}.intersection(sys.modules)
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
