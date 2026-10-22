"""Reader-option and optional-dependency checks without importing scientific libraries."""

from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

from super_deli import load_config, validate_dataset, write_config
from super_deli.core.schema import ConfigError, config_from_dict
from super_deli.io.genomic import GenomicIngestionError, ingest_genomic

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "genomic" / "dataset.json"


def test_reader_options_round_trip_preserves_codes_and_explicit_missing_states(tmp_path):
    payload = load_config(EXAMPLE).to_dict()
    payload["modalities"][0].update(
        format="structure",
        path="tiny_one_row.str",
        reader_options={"allele_encoding": {"1": "A", "2": "C", "-9": "N"}},
    )
    config = config_from_dict(payload)
    output = tmp_path / "dataset.json"
    write_config(config, output)
    assert load_config(output) == config
    assert config.to_dict()["modalities"][0]["reader_options"]["allele_encoding"]["-9"] == "N"


@pytest.mark.parametrize(
    "options",
    [
        {"has_popids": "false"},
        {"has_marker_names": 1},
        {"allele_start_col": True},
        {"unrecognized_option": 1},
        {"allele_encoding": ["A", "C"]},
    ],
)
def test_reader_option_types_are_strict(options):
    payload = load_config(EXAMPLE).to_dict()
    payload["modalities"][0]["reader_options"] = options
    with pytest.raises(ConfigError):
        config_from_dict(payload)


@pytest.mark.parametrize(
    "format,options",
    [
        ("phylip", {"has_popids": True}),
        ("vcf", {"allele_encoding": {"1": "A"}}),
        ("structure", {"allele_encoding": {"1": "invalid"}}),
        ("structure", {"allele_start_col": 0}),
        ("structure", {"allele_start_col": 1, "has_popids": True}),
    ],
)
def test_inapplicable_or_invalid_options_are_reported(format, options):
    payload = load_config(EXAMPLE).to_dict()
    payload["modalities"][0].update(format=format, reader_options=options)
    report = validate_dataset(config_from_dict(payload), base_dir=EXAMPLE.parent, check_files=False)
    assert not report.valid
    assert any(issue.code == "invalid_reader_options" for issue in report.issues)


def test_missing_dependency_has_install_guidance_and_creates_no_outputs(tmp_path, monkeypatch):
    def missing_version(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr("super_deli.io.genomic.version", missing_version)
    output = tmp_path / "run"
    with pytest.raises(GenomicIngestionError, match="super-deli\\[genomics\\]"):
        ingest_genomic(load_config(EXAMPLE), base_dir=EXAMPLE.parent, output_dir=output)
    assert not output.exists()
