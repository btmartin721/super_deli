"""Coordinate declarations are strict without changing existing configurations."""

from pathlib import Path

import pytest

from super_deli import MorphometricReaderOptions, load_config, validate_dataset, write_config
from super_deli.core.provenance import configuration_hash
from super_deli.core.schema import ConfigError, config_from_dict

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def payload():
    return load_config(EXAMPLES / "morphometric/dataset.json").to_dict()


def test_options_round_trip(tmp_path):
    config = config_from_dict(payload())
    write_config(config, tmp_path / "config.json")
    assert load_config(tmp_path / "config.json") == config
    assert isinstance(config.modalities[0].morphometric_options, MorphometricReaderOptions)


@pytest.mark.parametrize(
    "name,digest",
    [
        ("foundation", "a9e974fe5e7028cdb0cd511ea3a2ca14d4bb197dd87b466b4974582c525ef9ea"),
        ("genomic", "5cb6f966ac44c4880201c5d41cb4e9e2f0d55f2c0422fbc073da11a2e411df18"),
        ("baseline", "0fe6e52aea756e1fb9b88ae8b179df6a9d8b4a29221a0c443ae91cd6e528b628"),
    ],
)
def test_existing_configuration_hashes_remain_unchanged(name, digest):
    document = load_config(EXAMPLES / name / "dataset.json").to_dict()
    assert all("morphometric_options" not in m for m in document["modalities"])
    assert configuration_hash(document) == digest


@pytest.mark.parametrize(
    "options",
    [
        [],
        {},
        {"dimensions": 2},
        {"dimensions": True, "landmark_ids": ["a"]},
        {"dimensions": 2.0, "landmark_ids": ["a"]},
        {"dimensions": 2, "landmark_ids": "a"},
        {"dimensions": 2, "landmark_ids": [""]},
        {"dimensions": 2, "landmark_ids": ["a"], "unknown": 1},
        {"dimensions": 2, "landmark_ids": ["a"], "missing_landmark": [True, 0]},
        {"dimensions": 2, "landmark_ids": ["a"], "missing_landmark": [float("inf"), 0]},
        {"dimensions": 2, "landmark_ids": ["a"], "coordinate_unit": ""},
        {"dimensions": 2, "landmark_ids": ["a"], "missing_values": "NA"},
    ],
)
def test_option_field_types_are_strict(options):
    document = payload()
    document["modalities"][0]["morphometric_options"] = options
    with pytest.raises((ConfigError, ValueError)):
        config_from_dict(document)


@pytest.mark.parametrize(
    "changes",
    [
        {"dimensions": 1},
        {"dimensions": 4},
        {"landmark_ids": []},
        {"landmark_ids": ["a", "a"]},
        {"missing_landmark": [-1]},
        {"tps_id_field": "auto"},
        {"missing_values": ["NA", "NA"]},
    ],
)
def test_invalid_geometry_is_reported(changes):
    document = payload()
    document["modalities"][0]["morphometric_options"].update(changes)
    assert not validate_dataset(
        config_from_dict(document), base_dir=EXAMPLES / "morphometric", check_files=False
    ).valid


def test_options_are_scoped_to_coordinate_sources():
    for kind, format in (("genomic", "phylip"), ("morphometric", "image_manifest")):
        document = payload()
        document["modalities"][0].update(kind=kind, format=format)
        report = validate_dataset(
            config_from_dict(document), base_dir=EXAMPLES / "morphometric", check_files=False
        )
        assert not report.valid
        assert any(i.code == "invalid_morphometric_options" for i in report.issues)
    document = payload()
    document["modalities"][1]["morphometric_options"]["tps_id_field"] = "IMAGE"
    assert not validate_dataset(
        config_from_dict(document), base_dir=EXAMPLES / "morphometric", check_files=False
    ).valid
