"""Dependency-free reference-label boundaries and opt-in configuration."""

import subprocess
import sys

import pytest

from super_deli.io.reference_labels import read_reference_labels, validate_reference_labels
from super_deli.models.residual_config import ResidualConfig


def config():
    return ResidualConfig(
        "morphology", "ecology", ("m1", "m2", "m3"), ("e1", "e2", "e3"), True, "individual"
    )


def test_configuration_roundtrip_and_optional_imports():
    c = config()
    assert ResidualConfig.from_dict(c.to_dict()) == c
    code = (
        "import sys;from super_deli.models.residual_config import ResidualConfig;"
        "from super_deli.workflows.residual_cli import register;"
        "assert not {'torch','numpy','snpio'} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    "update",
    [
        {"schema_version": 2},
        {"random_seed": True},
        {"model_family": "other"},
        {"ecology_measurement_scope": "locality"},
        {"assume_independent_units": False},
        {"unknown": 1},
        {"morphology_features": ["reference_label", "a", "b"]},
    ],
)
def test_invalid_configuration(update):
    with pytest.raises(ValueError):
        ResidualConfig.from_dict({**config().to_dict(), **update})


def test_label_identity_and_class_order(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("unit_id,reference_label\nu3,b\nu1,a\nu4,b\nu2,a\n")
    labels = read_reference_labels(p)
    assert validate_reference_labels(["u4", "u1", "u3", "u2"], labels) == (
        ("a", "b"),
        (1, 0, 1, 0),
        {"a": 2, "b": 2},
    )


@pytest.mark.parametrize(
    "text",
    [
        "unit_id,reference_label,other\na,x,z\n",
        "unit_id,unit_id\na,b\n",
        "unit_id,reference_label\na,x\na,y\n",
        "unit_id,reference_label\na,\n",
        "unit_id,reference_label\n",
    ],
)
def test_malformed_label_csv(tmp_path, text):
    p = tmp_path / "labels.csv"
    p.write_text(text)
    with pytest.raises(ValueError):
        read_reference_labels(p)


@pytest.mark.parametrize(
    "labels",
    [
        {"a": "x"},
        {"a": "x", "b": "x", "c": "x", "d": "x"},
        {"a": "x", "b": "x", "c": "x", "d": "y"},
        {"a": "x", "b": "x", "c": " y", "d": "y"},
    ],
)
def test_label_coverage_or_insufficient_classes(labels):
    with pytest.raises(ValueError):
        validate_reference_labels(["a", "b", "c", "d"], labels)
