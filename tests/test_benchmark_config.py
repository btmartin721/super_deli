"""Dependency-free study declarations and stable random stream identities."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from super_deli.benchmarking.config import Arm, BenchmarkConfig, Scenario, derived_seed

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/benchmarking/smoke.json"


def declaration():
    return json.loads(EXAMPLE.read_text())


def test_config_roundtrip_and_dependency_free_import(tmp_path):
    config = BenchmarkConfig.from_dict(declaration())
    assert BenchmarkConfig.from_dict(config.to_dict()) == config
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from super_deli.benchmarking.config import BenchmarkConfig; "
            "assert not {'numpy', 'sklearn', 'snpio'} & sys.modules.keys()",
        ],
        check=True,
        cwd=tmp_path,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("n_datasets", True),
        ("n_units", 19),
        ("n_units", 22),
        ("n_loci", 7),
        ("random_seed", -1),
        ("random_seed", 2**64),
        ("max_group_members", 1),
        ("n_units", 10**8),
        ("unknown", 1),
    ],
)
def test_invalid_study_fields(field, value):
    options = declaration()
    options[field] = value
    with pytest.raises(ValueError):
        BenchmarkConfig.from_dict(options)


@pytest.mark.parametrize(
    "options",
    [
        {"name": "../bad"},
        {"name": "A"},
        {"name": "null", "pattern": "null"},
        {"name": "a", "strength": float("nan")},
        {"name": "a", "cell_missing_rate": 1},
        {"name": "a", "view_missing_rate": -0.1},
        {"name": "a", "unequal_groups": True},
        {"name": "a", "pattern": "unknown"},
        {"name": "a", "linkage": "guessed"},
        {"name": "a", "strength": True},
        {"name": "a", "strength": 10**500},
        {"name": "a", "extra": 1},
    ],
)
def test_invalid_scenario(options):
    with pytest.raises(ValueError):
        Scenario.from_dict(options)


@pytest.mark.parametrize(
    "kinds", [["genomic"], ["genomic", "genomic"], ["genomic", "bad"], "genomic"]
)
def test_invalid_arms(kinds):
    with pytest.raises(ValueError):
        Arm.from_dict({"name": "a", "kinds": kinds})


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_scenario",
        "duplicate_arm",
        "template_seed",
        "template_aggregation",
        "bootstrap_disabled",
        "no_shape",
    ],
)
def test_invalid_template_and_duplicate_names(change):
    options = declaration()
    if change == "duplicate_scenario":
        options["scenarios"].append(options["scenarios"][0])
    elif change == "duplicate_arm":
        options["arms"].append(options["arms"][0])
    elif change == "template_seed":
        options["model"]["random_seed"] = 2
    elif change == "template_aggregation":
        options["model"]["views"][0]["aggregation"] = "mean"
    elif change == "bootstrap_disabled":
        options["model"]["n_bootstrap"] = 0
    else:
        options["model"]["views"].pop()
    with pytest.raises(ValueError):
        BenchmarkConfig.from_dict(options)


def test_stream_identity_independent_of_iteration_order_or_study_size():
    keys = [(s, r, p) for s in ("a", "b") for r in range(3) for p in ("simulation", "analysis")]
    first = {k: derived_seed(42, *k) for k in keys}
    second = {k: derived_seed(42, *k) for k in reversed(keys)}
    assert first == second and len(set(first.values())) == len(keys)
    assert first[("a", 0, "simulation")] != derived_seed(43, "a", 0, "simulation")
