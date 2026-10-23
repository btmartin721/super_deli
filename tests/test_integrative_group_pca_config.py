"""The opt-in grouped policy preserves default serialization and null boundaries."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from super_deli.io.config import read_json
from super_deli.models.grouped_config import GroupedCalibrationConfig
from super_deli.models.integrative_config import (
    IntegrativeConfig,
    ViewConfig,
    require_calibrated_pca,
)

ROOT = Path(__file__).resolve().parents[1]


def model():
    return GroupedCalibrationConfig.from_dict(
        read_json(ROOT / "examples/shared-grouped-calibration/calibration.json")
    ).model


def test_default_policy_roundtrip_and_legacy_canonical_bytes():
    prior = read_json(ROOT / "examples/grouped-recovery/results/source_study.json")["calibration"]
    config = GroupedCalibrationConfig.from_dict(prior)
    assert json.loads(json.dumps(config.to_dict())) == prior
    assert all("pca_fit" not in v for v in config.to_dict()["model"]["views"])
    original = model()
    assert original.views[0].pca_fit == "source_rows"
    new = replace(
        original, views=(replace(original.views[0], pca_fit="target_means"), original.views[1])
    )
    assert IntegrativeConfig.from_dict(new.to_dict()) == new
    assert new.to_dict()["views"][0]["pca_fit"] == "target_means"
    assert original.n_pcs == 2 and original.covariance_type == "tied"
    assert IntegrativeConfig(True, views=original.views).covariance_type == "full"


@pytest.mark.parametrize(
    "options",
    [
        {"pca_fit": []},
        {"pca_fit": True},
        {"pca_fit": "unknown"},
        {"pca_fit": "target_means"},
        {"pca_fit": "target_means", "aggregation": "mean", "missing_policy": "mean"},
        {"pca_fit": "target_means", "aggregation": "mean", "count_transform": "log1p"},
    ],
)
def test_invalid_pca_options(options):
    with pytest.raises(ValueError):
        ViewConfig("traits", "morphological", **options)


def test_new_policy_is_scoped_and_rejected_by_calibration():
    config = GroupedCalibrationConfig.from_dict(
        read_json(ROOT / "examples/shared-grouped-calibration/calibration.json")
    )
    new = replace(
        config.model,
        views=(replace(config.model.views[0], pca_fit="target_means"), config.model.views[1]),
    )
    with pytest.raises(ValueError, match="not yet supported"):
        replace(config, model=new)
    with pytest.raises(ValueError, match="not yet supported"):
        require_calibrated_pca(new)
    with pytest.raises(ValueError, match="population/locality"):
        replace(new, target_level="specimen")
    with pytest.raises(ValueError, match="morphology"):
        ViewConfig("ecology", "ecological", aggregation="mean", pca_fit="target_means")
