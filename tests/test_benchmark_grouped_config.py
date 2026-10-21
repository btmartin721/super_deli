"""Dependency-free policies and actionable optional-dependency failures."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from super_deli.models.grouped_config import GroupedCalibrationConfig, GroupedStudyConfig

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/grouped-calibration"


def policy():
    return GroupedCalibrationConfig.from_dict(
        json.loads((EXAMPLES / "calibration.json").read_text())
    )


def test_roundtrip_and_optional_import_boundary(tmp_path):
    c = policy()
    assert GroupedCalibrationConfig.from_dict(c.to_dict()) == c
    s = GroupedStudyConfig.from_dict(json.loads((EXAMPLES / "study.json").read_text()))
    assert GroupedStudyConfig.from_dict(s.to_dict()) == s
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys;import super_deli.models.grouped_config;"
            "assert not {'numpy','snpio','scipy','sklearn'}&sys.modules.keys()",
        ],
        cwd=tmp_path,
        check=True,
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("assume_gaussian_independent_groups", False),
        ("assume_noninformative_counts", 1),
        ("alpha", 0),
        ("alpha", float("nan")),
        ("alpha", 10**500),
        ("rank_tol", 0),
        ("gradient_tol", 1),
        ("n_simulations", True),
        ("n_simulations", 0),
        ("random_seed", 2**64),
        ("max_archive_bytes", 0),
        ("schema_version", True),
        ("max_iterations", 0),
    ],
)
def test_invalid_policy(key, value):
    with pytest.raises(ValueError):
        replace(policy(), **{key: value})


def test_unsupported_models_and_unknown_fields():
    c = policy()
    for m in (
        replace(c.model, target_level="specimen"),
        replace(c.model, n_bootstrap=1),
        replace(c.model, max_k=1),
        replace(c.model, missing_views="complete_cases"),
        replace(c.model, views=tuple(replace(v, aggregation="none") for v in c.model.views)),
    ):
        with pytest.raises(ValueError):
            replace(c, model=m)
    with pytest.raises(ValueError):
        GroupedCalibrationConfig.from_dict({**c.to_dict(), "species_test": True})
    for change in ({"n_units": 19}, {"n_units": True}, {"random_seed": -1}, {"n_datasets": 0}):
        with pytest.raises(ValueError):
            GroupedStudyConfig(c, **change)


@pytest.mark.parametrize(
    "command",
    [
        "calibrate-grouped-integrative",
        "benchmark-grouped-calibration",
        "analyze-grouped-calibration",
    ],
)
def test_missing_dependencies_are_actionable(tmp_path, command):
    code = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'numpy','scipy','sklearn','snpio'}:
            raise ImportError('intentionally unavailable')
sys.meta_path.insert(0,Block())
from super_deli.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    if command == "calibrate-grouped-integrative":
        args = [
            command,
            str(EXAMPLES.parent / "foundation/dataset.json"),
            "--calibration-config",
            str(EXAMPLES / "calibration.json"),
        ]
    elif command == "benchmark-grouped-calibration":
        args = [command, str(EXAMPLES / "smoke.json"), "--case", "independent_gaussian"]
    else:
        args = [command, str(tmp_path)]
    result = subprocess.run(
        [sys.executable, "-c", code, *args, "--output-dir", str(tmp_path / "unused")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "super-deli[clustering]" in result.stderr
    assert "Traceback" not in result.stderr and not (tmp_path / "unused").exists()
