"""Contract and CLI checks using tiny declarations, independent of scientific runtimes."""

import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import replace

import pytest

from super_deli import __version__, load_config, validate_dataset, write_config
from super_deli.cli import main
from super_deli.core.provenance import collect_provenance, fingerprint_file
from super_deli.core.schema import ConfigError, config_from_dict


@pytest.fixture
def payload(tmp_path):
    (tmp_path / "traits.tsv").write_text("record\tlength\nscan1\t4.2\n", encoding="utf-8")
    (tmp_path / "environment.tsv").write_text("record\ttemperature\nsite1\t22\n", encoding="utf-8")
    return {
        "schema_version": 1,
        "dataset_id": "illustrative",
        "random_seed": 42,
        "populations": ["p1"],
        "localities": ["site1", "site2"],
        "specimens": [
            {"specimen_id": "s1", "population_id": "p1", "locality_id": "site1"},
            {"specimen_id": "s2", "population_id": "p1", "locality_id": "site1"},
        ],
        "modalities": [
            {
                "name": "traits",
                "kind": "morphological",
                "level": "specimen",
                "format": "tsv",
                "path": "traits.tsv",
                "observations": [
                    {"observation_id": "scan1", "unit_id": "s1"},
                    {"observation_id": "scan2", "unit_id": "s1"},
                ],
                "features": [{"name": "length", "dtype": "continuous", "unit": "mm"}],
            },
            {
                "name": "environment",
                "kind": "ecological",
                "level": "locality",
                "format": "tsv",
                "path": "environment.tsv",
                "observations": [{"observation_id": "reading1", "unit_id": "site1"}],
                "features": [{"name": "temperature", "dtype": "continuous", "unit": "degC"}],
            },
        ],
    }


def validate(payload, tmp_path):
    return validate_dataset(config_from_dict(payload), base_dir=tmp_path)


def codes(report):
    return {issue.code for issue in report.issues if issue.severity == "error"}


def test_round_trip_preserves_identifiers_types_and_paths(payload, tmp_path):
    config = config_from_dict(payload)
    output = tmp_path / "dataset.json"
    write_config(config, output)
    assert load_config(output) == config
    assert load_config(output).modalities[0].path == "traits.tsv"
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        write_config(replace(config, dataset_id="changed"), output)
    assert output.read_bytes() == before


def test_partial_coverage_and_replicates_are_counted_without_implied_matching(payload, tmp_path):
    report = validate(payload, tmp_path)
    assert report.valid
    traits, ecology = report.coverage
    assert (traits.observation_count, traits.unit_count, traits.repeated_observation_count) == (
        2,
        1,
        1,
    )
    assert traits.missing_unit_ids == ("s2",)
    assert (ecology.level, ecology.unit_count, ecology.missing_unit_ids) == (
        "locality",
        1,
        ("site2",),
    )
    assert any(i.code == "contents_not_checked" for i in report.issues)


@pytest.mark.parametrize(
    "kind,format",
    [
        ("genomic", "vcf"),
        ("genomic", "phylip"),
        ("genomic", "structure"),
        ("genomic", "genepop"),
        ("morphometric", "tps"),
        ("morphometric", "image_manifest"),
        ("morphometric", "landmarks_csv"),
        ("morphological", "csv"),
        ("ecological", "tsv"),
    ],
)
def test_each_declared_data_type_can_be_used_alone(payload, tmp_path, kind, format):
    modality = payload["modalities"][0]
    payload["modalities"] = [modality]
    modality.update(kind=kind, format=format)
    if kind in {"genomic", "morphometric"}:
        modality.pop("features")
    config = config_from_dict(payload)
    assert config.modalities[0].popmapfile is None
    assert validate_dataset(config, base_dir=tmp_path).valid


def test_different_specimens_can_share_group_links_without_being_merged(payload, tmp_path):
    payload["modalities"][1].update(
        level="specimen", observations=[{"observation_id": "other_record", "unit_id": "s2"}]
    )
    report = validate(payload, tmp_path)
    assert report.valid
    assert [c.missing_unit_ids for c in report.coverage] == [("s2",), ("s1",)]


def test_group_level_only_dataset_does_not_require_specimens(payload, tmp_path):
    payload["specimens"] = []
    payload["modalities"] = payload["modalities"][1:]
    assert validate(payload, tmp_path).valid


@pytest.mark.parametrize(
    "mutation,expected",
    [
        (lambda p: p.update(modalities=[]), "no_modalities"),
        (lambda p: p["specimens"].append(p["specimens"][0]), "duplicate_id"),
        (lambda p: p["populations"].append("p1"), "duplicate_id"),
        (lambda p: p["modalities"].append(copy.deepcopy(p["modalities"][0])), "duplicate_id"),
        (
            lambda p: p["modalities"][0]["observations"].append(
                p["modalities"][0]["observations"][0]
            ),
            "duplicate_id",
        ),
        (lambda p: p["specimens"][0].update(population_id="absent"), "unknown_unit"),
        (lambda p: p["modalities"][0]["observations"][0].update(unit_id="S1"), "unknown_unit"),
        (lambda p: p["modalities"][0].update(observations=[]), "no_observations"),
        (lambda p: p["modalities"][0].update(kind="mystery"), "unsupported_kind"),
        (lambda p: p["modalities"][0].update(format="xlsx"), "unsupported_format"),
        (lambda p: p["modalities"][0].update(level="species"), "unsupported_level"),
        (lambda p: p["modalities"][0].update(popmapfile="a.popmap"), "invalid_popmap"),
        (lambda p: p["modalities"][0].update(features=[]), "no_features"),
        (
            lambda p: p["modalities"][0]["features"][0].update(dtype="integerish"),
            "unsupported_dtype",
        ),
        (lambda p: p["modalities"][0]["features"][0].update(role="label"), "unsupported_role"),
        (lambda p: p["modalities"][0]["features"][0].update(dtype="ordinal"), "missing_categories"),
        (
            lambda p: p["modalities"][0]["features"][0].update(categories=["small"]),
            "invalid_categories",
        ),
    ],
)
def test_inconsistent_declarations_have_structured_errors(payload, tmp_path, mutation, expected):
    mutation(payload)
    report = validate(payload, tmp_path)
    assert not report.valid
    assert expected in codes(report)


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"random_seed": True},
        {"random_seed": -1},
        {"random_seed": 1.5},
        {"dataset_id": " padded "},
        {"typo": "value"},
        {"modalities": "traits"},
        {"specimens": [None]},
    ],
)
def test_structural_errors_are_rejected(payload, change):
    payload.update(change)
    with pytest.raises(ConfigError):
        config_from_dict(payload)


@pytest.mark.parametrize(
    "text",
    [
        '{"schema_version": 1, "schema_version": 1}',
        '{"schema_version": 1, "random_seed": NaN}',
        '{"schema_version": 1, "random_seed": Infinity}',
        '{"schema_version":',
        "[]",
    ],
)
def test_invalid_or_ambiguous_json_is_rejected(tmp_path, text):
    source = tmp_path / "bad.json"
    source.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(source)


def test_declared_missing_files_are_reported_without_dropping_records(payload, tmp_path):
    payload["modalities"][0]["path"] = "missing.tsv"
    report = validate(payload, tmp_path)
    assert "unreadable_file" in codes(report)
    assert report.coverage[0].observation_count == 2


def test_optional_popmap_is_checked_when_present(payload, tmp_path):
    modality = payload["modalities"][0]
    modality.update(kind="genomic", format="vcf", features=[], popmapfile="missing.popmap")
    report = validate(payload, tmp_path)
    assert "unreadable_file" in codes(report)
    (tmp_path / "missing.popmap").write_text("scan1 p1\nscan2 p1\n", encoding="utf-8")
    assert validate(payload, tmp_path).valid


def test_feature_categories_preserve_order_and_covariates(payload, tmp_path):
    payload["modalities"][0]["features"].extend(
        [
            {
                "name": "stage",
                "dtype": "ordinal",
                "categories": ["juvenile", "adult"],
                "role": "covariate",
            },
            {"name": "pattern", "dtype": "binary", "categories": ["absent", "present"]},
            {"name": "color", "dtype": "categorical", "categories": ["blue", "green"]},
            {"name": "scales", "dtype": "count"},
        ]
    )
    assert validate(payload, tmp_path).valid
    config = config_from_dict(payload)
    assert config.modalities[0].features[1].categories == ("juvenile", "adult")


def test_validation_does_not_touch_inputs_or_create_outputs(payload, tmp_path):
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in tmp_path.iterdir()}
    assert validate(payload, tmp_path).valid
    assert before == {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in tmp_path.iterdir()
    }


def test_hashes_record_bytes_and_change_with_configuration_or_inputs(payload, tmp_path):
    config = config_from_dict(payload)
    first = collect_provenance(config, base_dir=tmp_path, hash_inputs=True)
    second = collect_provenance(config, base_dir=tmp_path, hash_inputs=True)
    assert first["configuration_sha256"] == second["configuration_sha256"]
    traits = tmp_path / "traits.tsv"
    expected = hashlib.sha256(traits.read_bytes()).hexdigest()
    assert fingerprint_file(traits).sha256 == expected
    traits.write_text("changed input\n", encoding="utf-8")
    assert fingerprint_file(traits).sha256 != expected
    changed = collect_provenance(replace(config, random_seed=99), base_dir=tmp_path)
    assert changed["configuration_sha256"] != first["configuration_sha256"]
    assert changed["input_hash_status"] == "not_requested"
    assert changed["inputs"] == []


def test_cli_resolves_references_from_config_parent_and_preserves_output(
    payload, tmp_path, capsys, monkeypatch
):
    source = tmp_path / "dataset.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    report = tmp_path / "report.json"
    monkeypatch.chdir(tmp_path.parent)
    assert main(["validate", str(source), "--report", str(report), "--hash-inputs"]) == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout == json.loads(report.read_text())
    assert stdout["provenance"]["input_hash_status"] == "computed"
    assert len(stdout["provenance"]["inputs"]) == 2
    before = report.read_bytes()
    assert main(["validate", str(source), "--report", str(report)]) == 2
    assert report.read_bytes() == before


def test_cli_invalid_dataset_has_nonzero_exit_and_does_not_hash(payload, tmp_path, capsys):
    payload["modalities"][0]["path"] = "missing.tsv"
    source = tmp_path / "dataset.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    assert main(["validate", str(source), "--hash-inputs"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert not report["valid"]
    assert report["provenance"]["input_hash_status"] == "skipped_invalid_dataset"


def test_cli_malformed_config_has_clear_error(tmp_path, capsys):
    source = tmp_path / "dataset.json"
    source.write_text("{}", encoding="utf-8")
    assert main(["validate", str(source)]) == 2
    assert "schema_version" in capsys.readouterr().err


def test_module_entry_point_runs_outside_source_checkout(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "super_deli", "--version"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert __version__ in completed.stdout


def test_hashing_rejects_input_that_changes_during_read(tmp_path, monkeypatch):
    source = tmp_path / "changing.tsv"
    source.write_bytes(b"original")
    real_sha256 = hashlib.sha256

    class ChangingDigest:
        def __init__(self):
            self.digest = real_sha256()
            self.changed = False

        def update(self, chunk):
            self.digest.update(chunk)
            if not self.changed:
                self.changed = True
                source.write_bytes(b"modified")

        def hexdigest(self):
            return self.digest.hexdigest()

    monkeypatch.setattr("super_deli.core.provenance.hashlib.sha256", ChangingDigest)
    with pytest.raises(OSError, match="changed while hashing"):
        fingerprint_file(source)


def test_no_modality_dependency_is_imported():
    assert "snpio" not in sys.modules
    assert "torch" not in sys.modules
