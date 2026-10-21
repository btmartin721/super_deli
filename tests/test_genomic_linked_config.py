"""Dependency-free block-map policies and hypothesis boundaries."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from super_deli.models.linked_config import LinkedCalibrationConfig, LinkedStudyConfig

ROOT = Path(__file__).resolve().parents[1]


def policy():
    return LinkedCalibrationConfig.from_dict(
        json.loads((ROOT / "examples/linked-calibration/calibration.json").read_text())
    )


def test_roundtrips_and_optional_import_boundary(tmp_path):
    c = policy()
    assert LinkedCalibrationConfig.from_dict(c.to_dict()) == c
    study = LinkedStudyConfig.from_dict(
        json.loads((ROOT / "examples/linked-calibration/study.json").read_text())
    )
    assert LinkedStudyConfig.from_dict(study.to_dict()) == study
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys;from super_deli.models.linked_config import LinkedCalibrationConfig;"
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
        "assume_independent_blocks",
        "assume_fixed_panel",
        "assume_prespecified_blocks",
    ],
)
@pytest.mark.parametrize("value", [False, 1])
def test_scientific_assumptions_are_explicit(field, value):
    with pytest.raises(ValueError):
        replace(policy(), **{field: value})


@pytest.mark.parametrize(
    "change",
    [
        {"block_ids": ("one",) * 128},
        {"block_ids": ("a", " b")},
        {"block_ids": ["a", "b"]},
        {"n_simulations": 0},
        {"n_simulations": True},
        {"random_seed": -1},
        {"alpha": float("nan")},
        {"schema_version": 2},
    ],
)
def test_invalid_policies(change):
    with pytest.raises(ValueError):
        replace(policy(), **change)


@pytest.mark.parametrize(
    "change",
    [
        {"assume_unlinked": True},
        {"missing_policy": "mean"},
        {"n_bootstrap": 1},
        {"max_k": 1},
        {"scaling": "center"},
    ],
)
def test_model_must_match_complete_linked_scope(change):
    c = policy()
    with pytest.raises(ValueError):
        replace(c, model=replace(c.model, **change))


def test_unknown_null_assumptions_and_block_array_types():
    c = policy().to_dict()
    with pytest.raises(ValueError):
        LinkedCalibrationConfig.from_dict({**c, "assume_hardy_weinberg": True})
    with pytest.raises(ValueError):
        LinkedCalibrationConfig.from_dict({**c, "block_ids": "chr1"})
