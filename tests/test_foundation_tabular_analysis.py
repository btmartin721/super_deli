"""Strict comparison configuration and dependency-free partition accounting."""

from dataclasses import replace

import pytest

from super_deli.core.partition_comparison import compare_partitions
from super_deli.models.hdbscan_config import HDBSCANConfig
from super_deli.models.tabular_analysis_config import TabularAnalysisArm, TabularAnalysisConfig
from super_deli.models.tabular_cluster_config import TabularClusterConfig


def config():
    return TabularAnalysisConfig(
        (
            TabularAnalysisArm(
                "gmm", TabularClusterConfig(True, feature_names=("x", "y"), n_bootstrap=0)
            ),
            TabularAnalysisArm("density", HDBSCANConfig(True, feature_names=("x", "y"))),
        )
    )


def test_configuration_roundtrip_and_old_contract_isolation():
    c = config()
    assert TabularAnalysisConfig.from_dict(c.to_dict()) == c
    for cls in (TabularClusterConfig, HDBSCANConfig):
        with pytest.raises(ValueError):
            cls.from_dict(c.to_dict())
    assert TabularClusterConfig(True).n_bootstrap == 100
    assert HDBSCANConfig(True).n_subsamples == 0


@pytest.mark.parametrize("name", ["", "../escape", "A", "a/b", "a b", "x" * 65, None])
def test_unsafe_arm_names(name):
    with pytest.raises(ValueError):
        replace(config().arms[0], name=name)


@pytest.mark.parametrize(
    "change",
    [
        {"feature_names": ()},
        {"scaling": "center"},
        {"missing_policy": "mean"},
        {"count_transform": "log1p"},
        {"n_threads": 2},
    ],
)
def test_conflicting_gmm_preprocessing(change):
    with pytest.raises(ValueError):
        replace(config().arms[0], model=replace(config().arms[0].model, **change))


@pytest.mark.parametrize(
    "change", [{"n_pcs": 1}, {"feature_names": ("y", "x")}, {"rank_tol": 1e-9}]
)
def test_same_cohort_does_not_allow_different_transformations(change):
    c = config()
    with pytest.raises(ValueError):
        replace(c, arms=(c.arms[0], replace(c.arms[1], model=replace(c.arms[1].model, **change))))


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        [],
        {"schema_version": 1, "arms": [], "extra": 1},
        {"schema_version": True, "arms": []},
        {"schema_version": 2, "arms": []},
        {"schema_version": 1, "arms": []},
        {"schema_version": 1, "arms": "x"},
    ],
)
def test_invalid_analysis_contract(value):
    with pytest.raises(ValueError):
        TabularAnalysisConfig.from_dict(value)


def test_duplicate_and_many_arms():
    c = config()
    with pytest.raises(ValueError, match="Duplicate"):
        replace(c, arms=(c.arms[0], c.arms[0]))
    with pytest.raises(ValueError):
        replace(c, arms=tuple(replace(c.arms[0], name=f"m{i}") for i in range(17)))


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"name": "a", "family": [], "model": {}},
        {"name": "a", "family": "other", "model": {}},
        {"name": "a", "family": "gmm", "model": {}, "extra": 1},
    ],
)
def test_strict_arm(value):
    with pytest.raises(ValueError):
        TabularAnalysisArm.from_dict(value)


def test_label_invariance_noise_and_exact_denominators():
    result = compare_partitions([0, 0, 1, -1, 2], [4, 4, 9, 7, -1], [True] * 5, [True] * 5)
    assert result["joint_resolved"] == 3 and result["joint_pairs"] == 3
    assert result["joint_coverage"] == 0.6
    assert result["adjusted_rand"] == 1 and result["discordant_pairs"] == 0
    assert result["left_resolved"] == result["right_resolved"] == 4
    empty = compare_partitions([-1] * 3, [-1] * 3, [True] * 3, [True] * 3)
    assert empty["status"] == "undefined" and empty["adjusted_rand"] is None
    assert empty["joint_pairs"] == 0
    constant = compare_partitions([0] * 3, [7] * 3, [True] * 3, [True] * 3)
    assert constant["degenerate"] and constant["adjusted_rand"] == 1


@pytest.mark.parametrize(
    "args",
    [
        ([0], [0, 1], [True], [True]),
        ([True], [0], [True], [True]),
        ([0], [0], [1], [True]),
        ([-2], [0], [True], [True]),
    ],
)
def test_invalid_partitions(args):
    with pytest.raises(ValueError):
        compare_partitions(*args)


def test_frozen_validation_budget_and_disjoint_seed_streams():
    from super_deli.benchmarking.tabular_comparison import VALIDATION, tasks

    main, preflight = tasks("main"), tasks("preflight")
    assert len(main) == 36 and len(preflight) == 2
    assert sum(4 * (1 + t["draws"]) + 4 * int(t["replay"]) for t in main) == 672
    assert sum(4 * (1 + t["draws"]) for t in preflight) == 88
    assert VALIDATION["artifact_cap_bytes"] == 512 * 1024**2
    assert {tuple(t["data_seed"]) for t in main}.isdisjoint(
        tuple(t["data_seed"]) for t in preflight
    )


def test_validation_runner_import_does_not_spawn_or_parse_cli(monkeypatch):
    import runpy
    from pathlib import Path

    monkeypatch.setattr("sys.argv", ["runner", "--not-a-valid-argument"])
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples/tabular-analysis/validate.py"),
        run_name="import_only",
    )
    assert callable(namespace["main"])
