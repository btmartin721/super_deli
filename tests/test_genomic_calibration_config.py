"""Strict dependency-free assumptions, policy bounds and optional dependencies."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from super_deli.models.genomic_config import GenomicCalibrationConfig, GenomicStudyConfig

ROOT = Path(__file__).resolve().parents[1]


def policy():
    return GenomicCalibrationConfig.from_dict(
        json.loads((ROOT / "examples/genomic-calibration/calibration.json").read_text())
    )


def test_config_roundtrip_and_no_optional_imports(tmp_path):
    c = policy()
    assert GenomicCalibrationConfig.from_dict(c.to_dict()) == c
    study = GenomicStudyConfig.from_dict(
        json.loads((ROOT / "examples/genomic-calibration/study.json").read_text())
    )
    assert GenomicStudyConfig.from_dict(study.to_dict()) == study
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys;from super_deli.models.genomic_config import GenomicCalibrationConfig;"
                "assert not {'numpy','scipy','sklearn','snpio'}&sys.modules.keys()"
            ),
        ],
        cwd=tmp_path,
        check=True,
    )


@pytest.mark.parametrize(
    "field",
    [
        "assume_unrelated_specimens",
        "assume_hardy_weinberg",
        "assume_fixed_panel",
        "assume_noninformative_missingness",
    ],
)
@pytest.mark.parametrize("value", [False, 1])
def test_assumptions_are_explicit_booleans(field, value):
    with pytest.raises(ValueError, match="Explicitly"):
        replace(policy(), **{field: value})


@pytest.mark.parametrize(
    "change",
    [
        {"n_simulations": 0},
        {"n_simulations": True},
        {"random_seed": -1},
        {"alpha": float("nan")},
        {"alpha": 1},
        {"schema_version": 2},
        {"max_archive_bytes": 0},
        {"modality": " "},
    ],
)
def test_invalid_policies(change):
    with pytest.raises(ValueError):
        replace(policy(), **change)


@pytest.mark.parametrize(
    "change",
    [{"assume_unlinked": False}, {"n_bootstrap": 1}, {"max_k": 1}, {"scaling": "standard"}],
)
def test_model_must_match_the_declared_null_pipeline(change):
    c = policy()
    with pytest.raises(ValueError):
        replace(c, model=replace(c.model, **change))


def test_unknown_fields_and_imputation_in_study():
    c = policy()
    with pytest.raises(ValueError):
        GenomicCalibrationConfig.from_dict({**c.to_dict(), "maf_floor": 0.01})
    with pytest.raises(ValueError):
        GenomicStudyConfig(replace(c, model=replace(c.model, missing_policy="error")))
