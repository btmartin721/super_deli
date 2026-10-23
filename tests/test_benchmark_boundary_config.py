"""Versioned policies preserve schema-1 serialization and dependency boundaries."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from super_deli.models.grouped_config import (
    BoundarySolverConfig,
    GroupedCalibrationConfig,
    GroupedStudyConfig,
    study_cases,
)

ROOT = Path(__file__).resolve().parents[1]


def test_policy_roundtrip_legacy_bytes_and_optional_import_boundary(tmp_path):
    for folder in ("grouped-calibration", "boundary-calibration"):
        raw = json.loads((ROOT / f"examples/{folder}/calibration.json").read_text())
        c = GroupedCalibrationConfig.from_dict(raw)
        assert GroupedCalibrationConfig.from_dict(c.to_dict()) == c
        assert ("solver" in c.to_dict()) == (c.schema_version == 2)
        study = GroupedStudyConfig.from_dict(
            json.loads((ROOT / f"examples/{folder}/study.json").read_text())
        )
        assert GroupedStudyConfig.from_dict(study.to_dict()) == study
        assert len(study_cases(study)) == (7 if c.schema_version == 2 else 4)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys;from super_deli.models.grouped_config import BoundarySolverConfig;"
            "assert not {'numpy','scipy','sklearn','snpio'}&sys.modules.keys()",
        ],
        cwd=tmp_path,
        check=True,
    )


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2, "solver": None},
        {"schema_version": 1, "solver": BoundarySolverConfig()},
        {"schema_version": 3},
    ],
)
def test_invalid_versions(change):
    raw = json.loads((ROOT / "examples/boundary-calibration/calibration.json").read_text())
    c = GroupedCalibrationConfig.from_dict(raw)
    with pytest.raises(ValueError):
        replace(c, **change)


@pytest.mark.parametrize(
    "change",
    [
        {"estimator": "repair"},
        {"max_polish_iterations": True},
        {"max_polish_iterations": 0},
        {"max_line_search": 0},
    ],
)
def test_solver_policy_is_bounded_and_strict(change):
    with pytest.raises(ValueError):
        BoundarySolverConfig(**change)
    with pytest.raises(ValueError):
        BoundarySolverConfig.from_dict({"estimator": "conditional_psd_v1", "variance_floor": 1e-6})


def test_schema_two_prespecified_family_and_legacy_bytes():
    raw = json.loads((ROOT / "examples/boundary-calibration/calibration.json").read_text())
    full = GroupedCalibrationConfig.from_dict(raw)
    tied = replace(full, model=replace(full.model, covariance_type="tied"))
    assert GroupedCalibrationConfig.from_dict(tied.to_dict()) == tied
    assert "covariance_type" not in full.to_dict()["model"]
    assert tied.to_dict()["model"]["covariance_type"] == "tied"
    with pytest.raises(ValueError, match="legacy"):
        replace(tied, schema_version=1, solver=None)
