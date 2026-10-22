"""Strict tabular options and compatibility of existing version-one configurations."""

import copy
from pathlib import Path

import pytest

from super_deli import TabularReaderOptions, load_config, validate_dataset, write_config
from super_deli.core.schema import ConfigError, config_from_dict

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def payload():
    return load_config(EXAMPLES / "tabular" / "dataset.json").to_dict()


def test_options_roundtrip_and_existing_configs_stay_canonical(tmp_path):
    config = config_from_dict(payload())
    output = tmp_path / "config.json"
    write_config(config, output)
    assert load_config(output) == config
    assert config.modalities[0].tabular_options == TabularReaderOptions(
        "measurement_id", "specimen_id", ("", "NA"), ("notes",)
    )
    for directory in ("foundation", "genomic", "baseline"):
        original = load_config(EXAMPLES / directory / "dataset.json")
        assert all("tabular_options" not in m for m in original.to_dict()["modalities"])
        assert config_from_dict(original.to_dict()) == original


@pytest.mark.parametrize(
    "options",
    [
        [],
        {"unknown": 1},
        {"observation_id_column": ""},
        {"observation_id_column": 1},
        {"unit_id_column": False},
        {"missing_values": "NA"},
        {"missing_values": [None]},
        {"missing_values": [1]},
        {"missing_values": [" NA"]},
        {"ignored_columns": "notes"},
        {"ignored_columns": [""]},
    ],
)
def test_option_types_and_keys_are_strict(options):
    document = payload()
    document["modalities"][0]["tabular_options"] = options
    with pytest.raises(ConfigError):
        config_from_dict(document)


@pytest.mark.parametrize(
    "change",
    [
        {"observation_id_column": "length_mm"},
        {"unit_id_column": "measurement_id"},
        {"ignored_columns": ["notes", "notes"]},
        {"ignored_columns": ["sex"]},
        {"missing_values": ["", ""]},
        {"missing_values": ["striped"]},
    ],
)
def test_ambiguous_source_roles_and_missing_states_are_invalid(change):
    document = payload()
    document["modalities"][0]["tabular_options"].update(change)
    report = validate_dataset(
        config_from_dict(document), base_dir=EXAMPLES / "tabular", check_files=False
    )
    assert not report.valid
    assert any(i.code in {"duplicate_id", "ambiguous_missing_value"} for i in report.issues)


def test_tabular_options_are_rejected_for_other_modalities():
    document = load_config(EXAMPLES / "genomic" / "dataset.json").to_dict()
    document["modalities"][0]["tabular_options"] = {}
    report = validate_dataset(
        config_from_dict(document), base_dir=EXAMPLES / "genomic", check_files=False
    )
    assert not report.valid
    assert any(i.code == "invalid_tabular_options" for i in report.issues)


def test_default_identifier_column_cannot_also_be_a_feature():
    document = copy.deepcopy(payload())
    document["modalities"][1]["features"][0]["name"] = "observation_id"
    assert not validate_dataset(
        config_from_dict(document), base_dir=EXAMPLES / "tabular", check_files=False
    ).valid
