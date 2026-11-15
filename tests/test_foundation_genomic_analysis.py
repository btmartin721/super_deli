"""Strict dependency-free comparison declarations and lazy CLI."""

from dataclasses import replace

import pytest

from super_deli.models.config import BaselineConfig
from super_deli.models.genomic_analysis_config import GenomicAnalysisArm, GenomicAnalysisConfig
from super_deli.models.integrative_config import IntegrativeConfig, ViewConfig
from super_deli.models.tabular_cluster_config import TabularClusterConfig


def config():
    g = BaselineConfig(2, "error", True, n_bootstrap=0)
    t = TabularClusterConfig(True, n_bootstrap=0, feature_names=("x", "y"))
    views = (
        ViewConfig("snp", "genomic", ploidy=2),
        ViewConfig("traits", "morphological", feature_names=("x", "y")),
    )
    return GenomicAnalysisConfig(
        (
            GenomicAnalysisArm("genomic", "genomic", ("snp",), g),
            GenomicAnalysisArm("traits", "tabular", ("traits",), t),
            GenomicAnalysisArm(
                "combined",
                "integrative",
                ("snp", "traits"),
                IntegrativeConfig(True, views=views, n_bootstrap=0),
            ),
        )
    )


def test_roundtrip_and_legacy_bytes():
    c = config()
    assert GenomicAnalysisConfig.from_dict(c.to_dict()) == c
    assert "covariance_type" not in c.arms[-1].model.to_dict()
    assert replace(c.arms[-1].model, covariance_type="tied").to_dict()["covariance_type"] == "tied"


@pytest.mark.parametrize(
    "change", [{"schema_version": True}, {"arms": []}, {"ecology_measurement_scope": "locality"}]
)
def test_invalid_contract(change):
    with pytest.raises(ValueError):
        GenomicAnalysisConfig.from_dict({**config().to_dict(), **change})


def test_required_comparison_and_preprocessing():
    c = config()
    with pytest.raises(ValueError):
        replace(c, arms=c.arms[:2])
    with pytest.raises(ValueError):
        replace(c, arms=(c.arms[0], c.arms[1], replace(c.arms[-1], name="traits")))
    with pytest.raises(ValueError):
        replace(c, arms=(replace(c.arms[0], model=replace(c.arms[0].model, n_pcs=1)), *c.arms[1:]))
    with pytest.raises(ValueError):
        replace(c.arms[0], model=replace(c.arms[0].model, n_bootstrap=1))
    with pytest.raises(ValueError):
        replace(c.arms[1], block_ids=("a", "b"))
    with pytest.raises(ValueError):
        replace(c.arms[1], model=replace(c.arms[1].model, missing_policy="mean"))


def test_lazy_help(capsys):
    from super_deli.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["analyze-matched-genomic", "--help"])
    assert exc.value.code == 0
    assert "--analysis-config" in capsys.readouterr().out
