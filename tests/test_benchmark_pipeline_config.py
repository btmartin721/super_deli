"""Known-law assumptions, explicit designs and dependency-free declarations."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from super_deli.models.pipeline_config import PipelineCalibrationConfig, PipelineStudyConfig

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/pipeline-calibration"


def configuration():
    return PipelineCalibrationConfig.from_dict(
        json.loads((EXAMPLES / "calibration.json").read_text())
    )


def test_roundtrip_and_dependency_free_import(tmp_path):
    config = configuration()
    assert PipelineCalibrationConfig.from_dict(config.to_dict()) == config
    assert config.for_linkage(True).reference.unequal_groups
    study = PipelineStudyConfig.from_dict(json.loads((EXAMPLES / "study.json").read_text()))
    assert PipelineStudyConfig.from_dict(study.to_dict()) == study
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import super_deli.models.pipeline_config; "
            "assert not {'numpy', 'sklearn', 'snpio'} & sys.modules.keys()",
        ],
        cwd=tmp_path,
        check=True,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("assume_declared_reference", False),
        ("assume_declared_reference", 1),
        ("alpha", 0),
        ("alpha", float("nan")),
        ("alpha", 10**500),
        ("n_simulations", 0),
        ("n_simulations", True),
        ("random_seed", 2**64),
        ("max_archive_bytes", 0),
        ("schema_version", True),
    ],
)
def test_invalid_test_policies(field, value):
    with pytest.raises(ValueError):
        replace(configuration(), **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("law", "guessed_empirical_species"),
        ("n_units", 22),
        ("n_loci", 7),
        ("linkage", "guessed"),
        ("unequal_groups", True),
        ("cell_missing_rate", 1),
    ],
)
def test_invalid_population_designs(field, value):
    config = configuration()
    with pytest.raises(ValueError):
        replace(config, reference=replace(config.reference, **{field: value}))


def test_reject_mismatched_aggregation_bootstrap_and_unsafe_extra_options():
    config = configuration()
    for model in (
        replace(config.model, target_level="locality"),
        replace(config.model, n_bootstrap=1),
        replace(config.model, max_k=1),
    ):
        with pytest.raises(ValueError):
            replace(config, model=model)
    with pytest.raises(ValueError):
        PipelineCalibrationConfig.from_dict({**config.to_dict(), "fit_null_from_labels": True})


def test_audit_cli_reports_missing_optional_dependencies(tmp_path):
    code = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'sklearn', 'snpio', 'scipy'}:
            raise ImportError('optional dependency intentionally unavailable')
sys.meta_path.insert(0, Block())
from super_deli.cli import main
raise SystemExit(main(['analyze-pipeline-calibration', '.', '--output-dir', 'unused']))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "super-deli[clustering]" in result.stderr
    assert "Traceback" not in result.stderr
