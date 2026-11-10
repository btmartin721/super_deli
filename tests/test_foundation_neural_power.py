"""Dependency-free design, resource boundaries and CLI registration."""

import pytest

from super_deli.benchmarking.neural_power_config import NeuralPowerStudy, cells, counts, tasks


@pytest.mark.parametrize(
    "change",
    [
        dict(schema_version=True),
        dict(fixture=1),
        dict(unknown=1),
        dict(max_workers=3),
        dict(max_workers=True),
        dict(max_projected_seconds=36001),
        dict(preflight_max_seconds=7201),
        dict(artifact_cap_bytes=8 * 1024**3 + 1),
        dict(min_free_bytes=8 * 1024**3 - 1),
    ],
)
def test_reject_unapproved_design(change):
    with pytest.raises(ValueError):
        NeuralPowerStudy.from_dict(change)


def test_frozen_grid_and_scientific_budgets():
    c = NeuralPowerStudy()
    assert NeuralPowerStudy.from_dict(c.to_dict()) == c
    assert len(cells(c)) == 11
    assert len(tasks(c, "main")) == 100 and len(tasks(c, "preflight")) == 11
    assert counts(c, "main")["workflows"] == 20400
    assert counts(c, "preflight")["workflows"] == 2200
    assert counts(c, "main")["verified_updates"] == 10200000
    assert counts(c, "preflight")["verified_updates"] == 1100000
    assert len({t["key"] for t in tasks(c, "main")}) == 100
    paths = [
        tuple(t[name])
        for phase in ("main", "preflight")
        for t in tasks(c, phase)
        for name in (
            "data_seed",
            "noise_seed",
            "permutation_seed",
            "ecology_order_seed",
            "null_seed_path",
        )
    ]
    assert len(paths) == len(set(paths))
    assert all(
        t["n"] == 64
        for t in tasks(c, "main")
        if t["case"] in ("heavy_tail", "continuous_gradient", "unequal_complementary")
    )


def test_engineering_fixture_is_small_and_distinct():
    c = NeuralPowerStudy(fixture=True)
    assert counts(c, "main")["workflows"] == 12
    assert counts(c, "preflight")["workflows"] == 6
    assert tasks(c, "main")[0]["n"] == 32
    with pytest.raises(ValueError):
        tasks(c, "unknown")


def test_new_command_help_is_dependency_free(capsys):
    from super_deli.cli import main

    with pytest.raises(SystemExit) as e:
        main(["benchmark-neural-power", "--help"])
    assert e.value.code == 0
    assert "--staging-bytes" in capsys.readouterr().out
