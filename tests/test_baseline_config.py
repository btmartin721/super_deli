"""The model settings remain usable without optional scientific libraries."""

from dataclasses import replace

import pytest

from super_deli.io.config import read_json
from super_deli.models.config import BaselineConfig


def test_roundtrip():
    config = BaselineConfig(2, "error", True)
    assert BaselineConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    "change",
    [
        {"ploidy": 1},
        {"ploidy": True},
        {"n_pcs": 0},
        {"max_k": 0},
        {"min_cluster_size": 1},
        {"n_bootstrap": -1},
        {"n_init": False},
        {"schema_version": 2},
        {"missing_policy": "automatic"},
        {"missing_policy": []},
        {"scaling": {}},
        {"scaling": "guess"},
        {"assume_independent_specimens": False},
        {"assume_unlinked": 1},
        {"tol": float("nan")},
        {"reg_covar": 0},
        {"assignment_threshold": 1.1},
        {"random_seed": 2.0},
        {"n_threads": 0},
        {"max_pairwise_bytes": 0},
    ],
)
def test_invalid_settings(change):
    with pytest.raises(ValueError):
        replace(BaselineConfig(2, "error", True), **change)


@pytest.mark.parametrize("document", [[], {}, {"schema_version": 1, "unknown": 2}])
def test_malformed_settings(document):
    with pytest.raises(ValueError):
        BaselineConfig.from_dict(document)


def test_shared_strict_json(tmp_path):
    source = tmp_path / "settings.json"
    source.write_text('{"ploidy": 2, "ploidy": 1}')
    with pytest.raises(ValueError, match="duplicate"):
        read_json(source)
