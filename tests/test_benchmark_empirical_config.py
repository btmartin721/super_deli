"""Strict, dependency-free empirical null declarations."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from super_deli.models.empirical_config import EmpiricalCalibrationConfig, EmpiricalStudyConfig

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/empirical-calibration"


def configuration():
    return EmpiricalCalibrationConfig.from_dict(
        json.loads((EXAMPLES / "calibration.json").read_text())
    )


def test_roundtrip_and_optional_import_boundary(tmp_path):
    config = configuration()
    assert EmpiricalCalibrationConfig.from_dict(config.to_dict()) == config
    study = EmpiricalStudyConfig.from_dict(json.loads((EXAMPLES / "study.json").read_text()))
    assert EmpiricalStudyConfig.from_dict(study.to_dict()) == study
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import super_deli.models.empirical_config; "
            "assert not {'numpy','snpio','sklearn'} & sys.modules.keys()",
        ],
        cwd=tmp_path,
        check=True,
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("assume_iid_gaussian_raw_views", False),
        ("assume_iid_gaussian_raw_views", 1),
        ("alpha", 0),
        ("alpha", float("nan")),
        ("alpha", 10**500),
        ("rank_tol", 0),
        ("n_simulations", True),
        ("n_simulations", 0),
        ("random_seed", 2**64),
        ("max_archive_bytes", 0),
        ("schema_version", True),
    ],
)
def test_invalid_policies(name, value):
    with pytest.raises(ValueError):
        replace(configuration(), **{name: value})


def test_unsupported_analyses_and_unknown_fields():
    config = configuration()
    for model in (
        replace(config.model, n_bootstrap=1),
        replace(config.model, target_level="locality"),
        replace(config.model, missing_views="complete_cases"),
        replace(config.model, max_k=1),
        replace(
            config.model, views=tuple(replace(v, missing_policy="mean") for v in config.model.views)
        ),
    ):
        with pytest.raises(ValueError):
            replace(config, model=model)
    with pytest.raises(ValueError):
        EmpiricalCalibrationConfig.from_dict({**config.to_dict(), "infer_species": True})


@pytest.mark.parametrize(
    "command",
    [
        "calibrate-empirical-integrative",
        "benchmark-empirical-calibration",
        "analyze-empirical-calibration",
    ],
)
def test_missing_optional_dependencies_are_actionable(tmp_path, command):
    code = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy','sklearn','scipy','snpio'}:
            raise ImportError('intentionally unavailable')
sys.meta_path.insert(0,Block())
from super_deli.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    if command == "calibrate-empirical-integrative":
        args = [
            command,
            str(EXAMPLES.parent / "foundation/dataset.json"),
            "--calibration-config",
            str(EXAMPLES / "calibration.json"),
        ]
    elif command == "benchmark-empirical-calibration":
        args = [command, str(EXAMPLES / "smoke.json"), "--case", "spherical_gaussian"]
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
