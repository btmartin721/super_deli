"""Dependency-free scientific budgets and immutable legacy configuration."""

from dataclasses import replace

import pytest

from super_deli.benchmarking.neural_calibration_config import NeuralCalibrationStudy
from super_deli.benchmarking.neural_false_split_config import (
    NeuralFalseSplitStudy,
    batch_tasks,
    counts,
    tasks,
)


def test_approved_scientific_budget_and_balanced_batches():
    c = NeuralFalseSplitStudy()
    assert counts(c, "main")["workflows"] == 82400
    assert counts(c, "preflight")["workflows"] == 2800
    assert counts(c, "main")["verified_updates"] == 41200000
    assert counts(c, "preflight")["verified_updates"] == 1400000
    seen = []
    for i in range(8):
        batch = batch_tasks(c, i)
        assert len(batch) == 50
        assert sum(t["case"] == "gaussian_independent" for t in batch) == 25
        seen.extend(t["key"] for t in batch)
    assert len(set(seen)) == 400
    assert set(seen) == {t["key"] for t in tasks(c, "main")}


def test_draw_counts_and_disjoint_complete_entropy_paths():
    c = NeuralFalseSplitStudy()
    seeds = []
    for phase in ("preflight", "main"):
        grid = tasks(c, phase)
        for t in grid:
            assert t["n"] == 64
            for name, value in t.items():
                if name.endswith("seed") or name == "null_seed_path":
                    seeds.append(tuple(value))
    assert len(seeds) == len(set(seeds))
    assert all(t["n_simulations"] == 99 for t in tasks(c, "main"))
    assert sorted(t["n_simulations"] for t in tasks(c, "preflight")) == [99, 99, 99, 99, 999]


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("fixture", 1),
        ("max_workers", 3),
        ("max_workers", True),
        ("artifact_cap_bytes", 24 * 1024**3 + 1),
        ("max_projected_seconds", 144001),
        ("preflight_max_seconds", 7201),
        ("min_free_bytes", 8 * 1024**3 - 1),
        ("artifact_cap_bytes", 0),
    ],
)
def test_resource_policy_rejects_widening_and_wrong_types(field, value):
    with pytest.raises(ValueError):
        replace(NeuralFalseSplitStudy(), **{field: value})


def test_legacy_pilot_schema_and_resource_defaults_unchanged():
    old = NeuralCalibrationStudy()
    assert old.max_projected_seconds == 14400
    assert old.artifact_cap_bytes == 4 * 1024**3
    assert (
        NeuralFalseSplitStudy.from_dict(NeuralFalseSplitStudy().to_dict())
        == NeuralFalseSplitStudy()
    )
    with pytest.raises(ValueError):
        NeuralFalseSplitStudy.from_dict({"observations_per_law": 10})


def test_no_numpy_needed_to_register_new_command():
    from super_deli.cli import main

    with pytest.raises(SystemExit) as e:
        main(["benchmark-neural-false-splits", "--help"])
    assert e.value.code == 0
