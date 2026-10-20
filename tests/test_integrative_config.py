"""Strict, dependency-free integrative settings and cross-view declarations."""

import subprocess
import sys
from dataclasses import replace

import pytest

from super_deli.models.integrative_config import IntegrativeConfig, ViewConfig
from super_deli.models.shape_config import ShapeConfig


def config(**changes):
    return replace(
        IntegrativeConfig(
            True,
            views=(ViewConfig("traits", "morphological"), ViewConfig("environment", "ecological")),
        ),
        **changes,
    )


def test_roundtrip_with_all_view_kinds():
    model = config(
        views=(
            ViewConfig("genome", "genomic", ploidy=2),
            ViewConfig("traits", "morphological", feature_names=("length",)),
            ViewConfig("eco", "ecological"),
            ViewConfig("shape", "morphometric", shape=ShapeConfig()),
        )
    )
    assert IntegrativeConfig.from_dict(model.to_dict()) == model
    assert model.views[0].genomic_config().scaling == "patterson"
    assert model.views[1].tabular_config().scaling == "standard"


@pytest.mark.parametrize(
    "change",
    [
        {"assume_independent_units": False},
        {"schema_version": True},
        {"schema_version": 2},
        {"views": []},
        {"views": ()},
        {"views": (ViewConfig("one", "ecological"),)},
        {"target_level": "automatic"},
        {"target_level": []},
        {"missing_views": "impute"},
        {"max_working_bytes": 0},
        {"max_pairwise_bytes": 0},
        {"rank_tol": True},
        {"rank_tol": float("nan")},
        {"rank_tol": 1},
        {"n_pcs": 5, "min_cluster_size": 6},
        {"views": (ViewConfig("a", "ecological"), ViewConfig("b", "ecological"))},
    ],
)
def test_invalid_integrative_settings(change):
    with pytest.raises(ValueError):
        config(**change)


@pytest.mark.parametrize(
    "change",
    [
        {"modality": " "},
        {"kind": "image"},
        {"kind": []},
        {"n_pcs": 0},
        {"weight": True},
        {"weight": float("inf")},
        {"weight": 0},
        {"aggregation": []},
        {"aggregation": "auto"},
        {"ploidy": 2},
        {"shape": ShapeConfig()},
        {"feature_names": []},
        {"feature_names": ("a", "a")},
        {"feature_names": (" ",)},
        {"missing_policy": "auto"},
        {"scaling": False},
        {"count_transform": "auto"},
    ],
)
def test_invalid_tabular_view(change):
    with pytest.raises(ValueError):
        ViewConfig(**{"modality": "a", "kind": "morphological", **change})


@pytest.mark.parametrize(
    "view",
    [
        {"modality": "g", "kind": "genomic"},
        {"modality": "g", "kind": "genomic", "ploidy": True},
        {"modality": "g", "kind": "genomic", "ploidy": 2, "count_transform": "log1p"},
        {"modality": "g", "kind": "genomic", "ploidy": 2, "scaling": 0},
        {"modality": "s", "kind": "morphometric"},
        {"modality": "s", "kind": "morphometric", "shape": {"schema_version": 1}, "ploidy": 2},
        {
            "modality": "s",
            "kind": "morphometric",
            "shape": {"schema_version": 1, "repeated_policy": "allow"},
        },
        {
            "modality": "s",
            "kind": "morphometric",
            "shape": {"schema_version": 1, "n_components": 4},
        },
        {"modality": "a", "kind": "ecological", "unknown": 1},
    ],
)
def test_invalid_other_views(view):
    with pytest.raises(ValueError):
        ViewConfig.from_dict(view)


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"schema_version": 1},
        {"schema_version": 1, "views": [], "target_level": "specimen"},
    ],
)
def test_required_config_fields(value):
    with pytest.raises(ValueError):
        IntegrativeConfig.from_dict(value)


def test_imports_without_scientific_libraries():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from super_deli.models.integrative_config import IntegrativeConfig
from super_deli.workflows.integrative import run_integrative
assert not {'numpy','scipy','sklearn','snpio','PIL'} & sys.modules.keys()
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
