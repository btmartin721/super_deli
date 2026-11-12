"""Frozen design and dependency-free diagnostic registration."""

import os
import subprocess
import sys

import pytest

from super_deli.benchmarking.neural_power_config import NeuralPowerStudy
from super_deli.benchmarking.neural_power_config import tasks as power_tasks
from super_deli.benchmarking.weak_signal_config import WeakSignalStudy, counts_from_tasks, tasks


@pytest.mark.parametrize(
    "change",
    [
        dict(schema_version=True),
        dict(fixture=1),
        dict(unknown=1),
        dict(max_workers=3),
        dict(max_workers=True),
        dict(max_projected_seconds=1801),
        dict(preflight_max_seconds=601),
        dict(artifact_cap_bytes=1024**3 + 1),
        dict(min_free_bytes=8 * 1024**3 - 1),
    ],
)
def test_reject_unapproved_changes(change):
    with pytest.raises(ValueError):
        WeakSignalStudy.from_dict(change)


def test_exact_frozen_budgets():
    entries = [
        dict(task=t)
        for t in power_tasks(NeuralPowerStudy(), "main")
        if t["case"] in ("four_complementary", "gaussian_independent")
    ]
    c = WeakSignalStudy()
    assert WeakSignalStudy.from_dict(c.to_dict()) == c
    for phase, n, searches, truth, fits, starts in (
        ("preflight", 8, 4, 12, 28, 172),
        ("main", 80, 20, 120, 200, 920),
    ):
        grid = tasks(entries, phase)
        budget = counts_from_tasks(grid)
        assert len(grid) == n and budget["gmm_searches"] == searches
        assert budget["truth_diagnostics"] == truth
        assert budget["maximum_component_fits"] == fits
        assert budget["maximum_em_starts"] == starts
        assert budget["reference_evaluations"] == 3 * n
        assert budget["neural_fits"] == 0
        assert all(t["truth_fits"] == 0 for t in grid if t["case"] == "gaussian_independent")


def test_dependency_free_help():
    code = """import sys
class Block:
 def find_spec(self, name, *args):
  if name.split('.')[0] in {'numpy','scipy','sklearn','torch','snpio'}:raise AssertionError(name)
sys.meta_path.insert(0,Block())
from super_deli.cli import main
raise SystemExit(main(['diagnose-weak-signal','--help']))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=os.environ
    )
    assert result.returncode == 0, result.stderr
    assert "--archive-root" in result.stdout
