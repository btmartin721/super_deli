"""Dependency-free policy, compatibility and preset boundaries."""

from dataclasses import replace

import pytest

from super_deli.cli import main
from super_deli.io.config import read_json, write_json
from super_deli.models.full_retention import full_retention_analysis, full_retention_model
from super_deli.models.integrative_config import (
    IntegrativeConfig,
    ViewConfig,
    require_calibrated_pca,
)
from super_deli.models.multimodal_analysis_config import (
    MultimodalAnalysisArm,
    MultimodalAnalysisConfig,
)
from super_deli.models.shape_cluster_config import ShapeClusterConfig
from super_deli.models.shape_config import ShapeConfig
from super_deli.models.tabular_cluster_config import TabularClusterConfig


def config():
    return IntegrativeConfig(
        True,
        views=(
            ViewConfig("traits", "morphological", feature_names=("x", "y", "z")),
            ViewConfig("outline", "morphometric", shape=ShapeConfig()),
        ),
        covariance_type="tied",
        n_bootstrap=0,
    )


def analysis():
    model = config()
    return MultimodalAnalysisConfig(
        (
            MultimodalAnalysisArm(
                "traits",
                "tabular",
                ("traits",),
                TabularClusterConfig(True, feature_names=("x", "y", "z"), n_bootstrap=0),
            ),
            MultimodalAnalysisArm(
                "shape", "shape", ("outline",), ShapeClusterConfig(True, n_bootstrap=0)
            ),
            MultimodalAnalysisArm("combined", "integrative", ("traits", "outline"), model),
        )
    )


def test_legacy_serialization_and_explicit_dimension_remain_distinct():
    old = config()
    raw = old.to_dict()
    assert "fusion_policy" not in raw and old.n_pcs == 2
    assert IntegrativeConfig.from_dict(raw).to_dict() == raw
    explicit = replace(old, n_pcs=4)
    tagged = full_retention_model(old, min_cluster_size=5)
    assert tagged.n_pcs == 4 and tagged.fusion_policy == "retain_all_v1"
    assert tagged.to_dict() == {**explicit.to_dict(), "fusion_policy": "retain_all_v1"}
    assert IntegrativeConfig.from_dict(tagged.to_dict()) == tagged
    assert old.to_dict() == raw


@pytest.mark.parametrize(
    "changes",
    [
        {"fusion_policy": "automatic"},
        {"fusion_policy": True},
        {"n_pcs": 2},
        {"target_level": "population"},
        {"missing_views": "complete_cases"},
        {"covariance_type": "full"},
    ],
)
def test_tagged_policy_rejects_unsupported_scope(changes):
    tagged = full_retention_model(config(), min_cluster_size=5)
    with pytest.raises(ValueError):
        replace(tagged, **changes)


@pytest.mark.parametrize("minimum", [1, 4, True])
def test_minimum_is_never_silently_raised(minimum):
    with pytest.raises(ValueError):
        full_retention_model(config(), min_cluster_size=minimum)


def test_new_mode_has_no_inherited_significance():
    old = replace(config(), n_pcs=4)
    require_calibrated_pca(old)
    with pytest.raises(ValueError, match="Full-retention significance"):
        require_calibrated_pca(full_retention_model(old, min_cluster_size=5))


def test_preset_changes_only_selected_arm_and_retains_legacy_bytes():
    old = analysis()
    frozen = old.to_dict()
    new = full_retention_analysis(old, arm_name="combined", min_cluster_size=5)
    assert new.arms[:2] == old.arms[:2]
    assert new.arms[-1].model.n_pcs == 4
    assert old.to_dict() == frozen
    with pytest.raises(ValueError):
        full_retention_analysis(old, arm_name="shape", min_cluster_size=5)


def test_fit_free_cli_preserves_sources_and_existing_outputs(tmp_path, capsys):
    source, output = tmp_path / "analysis.json", tmp_path / "retained.json"
    write_json(analysis().to_dict(), source)
    before = source.read_bytes()
    args = [
        "full-retention-config",
        str(source),
        "--arm",
        "combined",
        "--min-cluster-size",
        "5",
        "--output-config",
        str(output),
    ]
    assert main(args) == 0
    assert '"fitting": "not_performed"' in capsys.readouterr().out
    assert source.read_bytes() == before
    assert MultimodalAnalysisConfig.from_dict(read_json(output)).arms[-1].model.n_pcs == 4
    retained = output.read_bytes()
    assert main(args) == 2
    assert output.read_bytes() == retained


def test_help_needs_no_numerical_dependencies(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["full-retention-config", "--help"])
    assert exc.value.code == 0
    assert "--min-cluster-size" in capsys.readouterr().out
