"""Dependency-free declarations, legacy schema preservation and lazy CLI."""

from dataclasses import replace

import pytest

from super_deli.models.integrative_config import IntegrativeConfig, ViewConfig
from super_deli.models.multimodal_analysis_config import (
    MultimodalAnalysisArm,
    MultimodalAnalysisConfig,
)
from super_deli.models.shape_cluster_config import ShapeClusterConfig
from super_deli.models.shape_config import ShapeConfig
from super_deli.models.tabular_cluster_config import TabularClusterConfig


def config():
    shape = ShapeClusterConfig(True, n_bootstrap=0)
    table = TabularClusterConfig(True, n_bootstrap=0, feature_names=("length", "depth"))
    views = (
        ViewConfig("outline", "morphometric", shape=shape.shape),
        ViewConfig("traits", "morphological", feature_names=table.feature_names),
    )
    combined = IntegrativeConfig(True, views=views, n_bootstrap=0)
    return MultimodalAnalysisConfig(
        (
            MultimodalAnalysisArm("shape", "shape", ("outline",), shape),
            MultimodalAnalysisArm("morphology", "tabular", ("traits",), table),
            MultimodalAnalysisArm("combined", "integrative", ("outline", "traits"), combined),
        )
    )


def test_roundtrip_without_genomics_and_legacy_rejection():
    from super_deli.models.genomic_analysis_config import GenomicAnalysisConfig

    c = config()
    assert MultimodalAnalysisConfig.from_dict(c.to_dict()) == c
    with pytest.raises(ValueError):
        GenomicAnalysisConfig.from_dict(c.to_dict())
    assert "covariance_type" not in c.arms[-1].model.to_dict()


@pytest.mark.parametrize(
    "change", [{"schema_version": True}, {"arms": []}, {"ecology_measurement_scope": "locality"}]
)
def test_malformed_config(change):
    with pytest.raises(ValueError):
        MultimodalAnalysisConfig.from_dict({**config().to_dict(), **change})


@pytest.mark.parametrize(
    "fault",
    [
        "drop_missing",
        "reflection_mismatch",
        "missing_singleton",
        "wrong_pcs",
        "unknown_family",
        "shape_blocks",
        "duplicate_arm",
        "grouped",
    ],
)
def test_boundaries(fault):
    c = config()
    arms = list(c.arms)
    with pytest.raises(ValueError):
        if fault == "drop_missing":
            arms[0] = replace(
                arms[0],
                model=replace(arms[0].model, shape=ShapeConfig(missing_policy="drop_observations")),
            )
        elif fault == "reflection_mismatch":
            arms[0] = replace(
                arms[0], model=replace(arms[0].model, shape=ShapeConfig(reflection="allow"))
            )
        elif fault == "missing_singleton":
            arms = arms[:1] + arms[2:]
        elif fault == "wrong_pcs":
            arms[0] = replace(arms[0], model=replace(arms[0].model, n_pcs=1))
        elif fault == "unknown_family":
            arms[0] = replace(arms[0], family="neural")
        elif fault == "shape_blocks":
            arms[0] = replace(arms[0], block_ids=("block",))
        elif fault == "duplicate_arm":
            arms[0] = replace(arms[0], name="combined")
        else:
            arms[-1] = replace(arms[-1], model=replace(arms[-1].model, target_level="population"))
        replace(c, arms=tuple(arms))


def test_equivalent_effective_shape_policy():
    c = config()
    a = c.arms[-1]
    v = replace(a.model.views[0], shape=ShapeConfig(n_components=2))
    c = replace(
        c, arms=(*c.arms[:2], replace(a, model=replace(a.model, views=(v, a.model.views[1]))))
    )
    assert c.arms[-1].model.views[0].shape.n_components == 2


def test_lazy_help(capsys):
    from super_deli.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["analyze-matched-multimodal", "--help"])
    assert exc.value.code == 0 and "--analysis-config" in capsys.readouterr().out
