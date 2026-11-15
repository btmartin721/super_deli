"""Opt-in declarations preserve historical schemas and fit-free behavior."""

from dataclasses import replace

import pytest

from super_deli.cli import main
from super_deli.io.config import read_json, write_json
from super_deli.models.full_retention import full_retention_analysis, full_retention_model
from super_deli.models.initialization import truth_free_analysis, truth_free_model
from super_deli.models.integrative_config import (
    IntegrativeConfig,
    ViewConfig,
    require_calibrated_pca,
)


def config():
    return IntegrativeConfig(
        True,
        views=(ViewConfig("m", "morphological"), ViewConfig("e", "ecological")),
        covariance_type="tied",
        n_bootstrap=0,
    )


def test_legacy_bytes_and_explicit_new_declaration():
    old = config()
    before = old.to_dict()
    assert "initialization_policy" not in before
    tagged = full_retention_model(old, min_cluster_size=5)
    new = truth_free_model(tagged)
    assert old.to_dict() == before and "initialization_policy" not in tagged.to_dict()
    assert new.to_dict() == {
        **tagged.to_dict(),
        "n_init": 100,
        "initialization_policy": "diverse_v1",
    }
    assert IntegrativeConfig.from_dict(new.to_dict()) == new
    with pytest.raises(ValueError, match="significance"):
        require_calibrated_pca(new)


@pytest.mark.parametrize(
    "changes",
    [
        {"initialization_policy": "auto"},
        {"initialization_policy": True},
        {"initialization_policy": {}},
        {"n_init": 10},
        {"fusion_policy": "joint_pca"},
    ],
)
def test_new_scope_is_explicit(changes):
    new = truth_free_model(full_retention_model(config(), min_cluster_size=5))
    with pytest.raises(ValueError):
        replace(new, **changes)


def test_unselected_arms_and_existing_files_are_preserved(tmp_path, capsys):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "initialization_config_fixture",
        Path(__file__).with_name("test_foundation_full_retention.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    analysis = module.analysis

    old = full_retention_analysis(analysis(), arm_name="combined", min_cluster_size=5)
    frozen = old.to_dict()
    new = truth_free_analysis(old, arm_name="combined")
    assert new.arms[:-1] == old.arms[:-1] and old.to_dict() == frozen
    with pytest.raises(ValueError):
        truth_free_analysis(old, arm_name="shape")
    src, out = tmp_path / "old.json", tmp_path / "new.json"
    write_json(frozen, src)
    before = src.read_bytes()
    args = ["truth-free-init-config", str(src), "--arm", "combined", "--output-config", str(out)]
    assert main(args) == 0 and src.read_bytes() == before
    assert read_json(out)["arms"][-1]["model"]["initialization_policy"] == "diverse_v1"
    after = out.read_bytes()
    assert main(args) == 2 and out.read_bytes() == after
    assert "not_performed" in capsys.readouterr().out


def test_roster_counts_and_distinct_fresh_seeds():
    from super_deli.benchmarking.initialization_config import InitializationStudyConfig, tasks

    main_tasks, pre = tasks("main"), tasks("preflight")
    assert len(main_tasks) == 200 and len(pre) == 10
    assert not {t["seed"] for t in main_tasks} & {t["seed"] for t in pre}
    assert sum(3 * (1 + 25 * t["stability"]) for t in main_tasks) + 30 == 2880
    assert sum(3 * (1 + 25 * t["stability"]) for t in pre) + 5 == 785
    config = InitializationStudyConfig("a" * 64, "b" * 64)
    assert InitializationStudyConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError):
        replace(config, artifact_cap_bytes=9 * 1024**3)


def test_help_is_dependency_free(capsys):
    with pytest.raises(SystemExit) as result:
        main(["truth-free-init-config", "--help"])
    assert result.value.code == 0 and "--output-config" in capsys.readouterr().out
