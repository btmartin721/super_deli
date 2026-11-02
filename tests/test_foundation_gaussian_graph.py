"""Dependency-light scope, budget, seed separation and installed help contracts."""

import subprocess
import sys

import pytest

from super_deli.benchmarking.gaussian_graph_config import StudyConfig, counts, draw_seed, tasks


def test_approved_budget_and_disjoint_streams():
    config = StudyConfig()
    assert counts(config, "main") == {
        "observations": 160,
        "tables": 320,
        "table_replays": 4,
        "evaluations": 97200,
        "point_outputs": 1920,
        "replay_outputs": 24,
        "embedding_step_bound": 1296,
    }
    assert counts(config, "preflight")["evaluations"] == 10800
    phases = [tasks(config, p) for p in ("preflight", "main")]
    assert {tuple(t["data_seed"]) for t in phases[0]}.isdisjoint(
        tuple(t["data_seed"]) for t in phases[1]
    )
    seeds = set()
    for phase in phases:
        for task in phase:
            for condition in ("clean", "noisy"):
                for role, count in (("pilot", 49), ("reference", 99)):
                    for i in range(count):
                        seed = tuple(draw_seed(task, condition, role, i))
                        assert seed not in seeds
                        seeds.add(seed)


@pytest.mark.parametrize(
    "change",
    [
        {"max_workers": 3},
        {"artifact_cap_bytes": 2**32},
        {"max_projected_seconds": 3601},
        {"fixture": 1},
    ],
)
def test_limits_cannot_expand(change):
    with pytest.raises(ValueError):
        StudyConfig(**change)


def test_unknown_seeds_and_phases_rejected():
    with pytest.raises(ValueError):
        tasks(StudyConfig(), "other")
    task = tasks(StudyConfig(), "main")[0]
    for condition, role, i in (("bad", "pilot", 0), ("clean", "bad", 0), ("clean", "pilot", 49)):
        with pytest.raises(ValueError):
            draw_seed(task, condition, role, i)


def test_module_help_without_scientific_imports():
    code = """
import sys
from super_deli.benchmarking.gaussian_graph_cli import main
try: main(["--help"])
except SystemExit as exc: assert exc.code == 0
assert not {"numpy", "scipy", "sklearn", "torch"}.intersection(sys.modules)
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
