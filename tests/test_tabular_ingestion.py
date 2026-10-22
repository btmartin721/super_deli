"""Typed-value truth, biological links, source safety and strict parsing boundaries."""

import csv
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from super_deli import TabularReaderOptions, load_config
from super_deli.cli import main
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.io.config import read_json
from super_deli.io.tabular import TabularIngestionError, ingest_tabular

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def snapshot(directory):
    return {
        str(p.relative_to(directory)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in directory.rglob("*")
        if p.is_file()
    }


@pytest.fixture
def source(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    shutil.copytree(EXAMPLES / "tabular", raw)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return raw, load_config(raw / "dataset.json")


def test_typed_truth_feature_order_repeats_partial_coverage_and_originals(source, tmp_path):
    raw, config = source
    before = snapshot(raw)
    loaded = ingest_tabular(config, base_dir=raw, output_dir=tmp_path / "run")
    expected = read_json(raw / "expected.json")
    assert [t.modality_name for t in loaded.tables] == [m.name for m in config.modalities]
    for table, summary in zip(loaded.tables, loaded.summary["modalities"], strict=True):
        truth = expected[table.modality_name]
        assert list(table.observation_ids) == truth["observation_ids"]
        assert list(table.unit_ids) == truth["unit_ids"]
        assert [f.name for f in table.features] == truth["feature_order"]
        assert [list(row) for row in table.values] == truth["values"]
        assert [list(row) for row in table.missing_mask] == truth["missing_mask"]
        saved = read_json(loaded.output_dir / summary["artifact"])
        assert saved["values"] == truth["values"] and saved["missing_mask"] == truth["missing_mask"]
        assert saved["features"][-1]["categories"] == list(table.features[-1].categories)
    traits, ecology = loaded.tables
    assert traits.features[0].unit == "mm" and traits.features[-1].role == "covariate"
    assert traits.features[3].categories == ("low", "intermediate", "high")
    assert type(traits.values[0][0]) is float and type(traits.values[0][1]) is int
    assert traits.values[-1][1] == 0 and traits.missing_mask[-1][1] is False
    assert ecology.level == "locality" and len(ecology.values) == 2  # never broadcast to specimens
    trait_coverage = loaded.summary["modalities"][0]["coverage"]
    assert trait_coverage["repeated_observation_count"] == 1
    assert trait_coverage["missing_unit_ids"] == ("specimen2",)
    assert loaded.summary["modalities"][1]["coverage"]["missing_unit_ids"] == ("site3",)
    assert loaded.summary["modalities"][0]["missing_count_by_column"]["sex"] == 1
    assert loaded.summary["provenance"]["configuration_sha256"] == configuration_hash(
        config.to_dict()
    )
    assert loaded.summary["provenance"]["input_hash_status"] == "computed_selected_modalities"
    for m, fingerprint in zip(
        config.modalities, loaded.summary["provenance"]["inputs"], strict=True
    ):
        assert fingerprint["sha256"] == fingerprint_file(raw / m.path).sha256
    assert snapshot(raw) == before and not list(Path.cwd().iterdir())
    assert not (loaded.output_dir / "failure.json").exists()
    assert not (loaded.output_dir / "result.json").exists()  # ingestion has no inference results


def test_existing_output_is_never_replaced(source, tmp_path):
    raw, config = source
    output = tmp_path / "existing"
    output.mkdir()
    (output / "sentinel").write_text("keep")
    before = snapshot(output)
    with pytest.raises(FileExistsError):
        ingest_tabular(config, base_dir=raw, output_dir=output)
    assert snapshot(output) == before


def make_source(
    tmp_path,
    rows,
    *,
    dtype="continuous",
    categories=(),
    options=None,
    header=None,
    format="csv",
    level="specimen",
    kind="morphological",
):
    path = tmp_path / f"input.{format}"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="," if format == "csv" else "\t")
        writer.writerow(header or ["observation_id", "value"])
        writer.writerows(rows)
    document = {
        "schema_version": 1,
        "dataset_id": "typed",
        "specimens": [{"specimen_id": "unit1"}] if level == "specimen" else [],
        "populations": ["unit1"] if level == "population" else [],
        "localities": ["unit1"] if level == "locality" else [],
        "modalities": [
            {
                "name": "table",
                "kind": kind,
                "level": level,
                "format": format,
                "path": path.name,
                "observations": [{"observation_id": "001", "unit_id": "unit1"}],
                "features": [{"name": "value", "dtype": dtype, "categories": list(categories)}],
            }
        ],
    }
    config = config_from_dict(document)
    if options is not None:
        config = replace(
            config, modalities=(replace(config.modalities[0], tabular_options=options),)
        )
    return path, config


@pytest.mark.parametrize("format", ["csv", "tsv"])
@pytest.mark.parametrize("kind", ["morphological", "ecological"])
@pytest.mark.parametrize("level", ["specimen", "population", "locality"])
def test_each_kind_level_and_format_is_standalone(tmp_path, format, kind, level):
    _, config = make_source(tmp_path, [["001", "-2.5e1"]], format=format, kind=kind, level=level)
    table = ingest_tabular(config, base_dir=tmp_path, output_dir=tmp_path / "run").tables[0]
    assert table.level == level and table.kind == kind
    assert table.observation_ids == ("001",) and table.values == ((-25.0,),)


@pytest.mark.parametrize(
    "value",
    [
        "NaN",
        "nan",
        "Inf",
        "-Infinity",
        "1e400",
        "1e-400",
        "1,000",
        "1_000",
        "12mm",
        " 2.5",
        "2.5 ",
        "NA",
        "٣",
    ],
)
def test_invalid_numeric_values_do_not_become_missing_or_silent_coercions(tmp_path, value):
    _, config = make_source(tmp_path, [["001", value]])
    with pytest.raises(TabularIngestionError, match="feature 'value'"):
        ingest_tabular(config, base_dir=tmp_path, output_dir=tmp_path / "run")
    assert read_json(tmp_path / "run" / "failure.json")["status"] == "failed"
    assert not (tmp_path / "run" / "summary.json").exists()


@pytest.mark.parametrize("value", ["-1", "1.0", "1e2", "+1", "True"])
def test_counts_require_nonnegative_integer_literals(tmp_path, value):
    _, config = make_source(tmp_path, [["001", value]], dtype="count")
    with pytest.raises(TabularIngestionError, match="count must"):
        ingest_tabular(config, base_dir=tmp_path, output_dir=tmp_path / "run")


def test_large_counts_remain_exact_and_numeric_category_labels_stay_strings(tmp_path):
    _, config = make_source(tmp_path, [["001", "9007199254740993"]], dtype="count")
    loaded = ingest_tabular(config, base_dir=tmp_path, output_dir=tmp_path / "count")
    assert loaded.tables[0].values == ((9007199254740993,),)
    assert read_json(loaded.output_dir / "modality_000/table.json")["values"] == [
        [9007199254740993]
    ]
    for dtype in ("binary", "ordinal", "categorical"):
        _, config = make_source(tmp_path, [["001", "01"]], dtype=dtype, categories=("01", "00"))
        assert ingest_tabular(config, base_dir=tmp_path, output_dir=tmp_path / dtype).tables[
            0
        ].values == (("01",),)


def test_missing_tokens_are_explicit_case_sensitive_and_do_not_apply_to_ids(tmp_path):
    options = TabularReaderOptions(missing_values=("", "NA", "001"))
    _, config = make_source(tmp_path, [["001", "NA"]], options=options)
    table = ingest_tabular(config, base_dir=tmp_path, output_dir=tmp_path / "run").tables[0]
    assert table.observation_ids == ("001",) and table.values == ((None,),)
    assert table.missing_mask == ((True,),)
    _, config = make_source(tmp_path, [["001", "na"]], options=options)
    with pytest.raises(TabularIngestionError):
        ingest_tabular(config, base_dir=tmp_path, output_dir=tmp_path / "lowercase")


def test_missing_looking_category_is_a_real_state_unless_declared_missing(tmp_path):
    _, config = make_source(
        tmp_path, [["001", "NA"]], dtype="categorical", categories=("NA", "plain")
    )
    assert ingest_tabular(config, base_dir=tmp_path, output_dir=tmp_path / "run").tables[
        0
    ].values == (("NA",),)


@pytest.mark.parametrize("dtype", ["binary", "ordinal", "categorical"])
def test_unlisted_category_state_is_an_error(tmp_path, dtype):
    _, config = make_source(tmp_path, [["001", "new"]], dtype=dtype, categories=("a", "b"))
    with pytest.raises(TabularIngestionError, match="unknown"):
        ingest_tabular(config, base_dir=tmp_path, output_dir=tmp_path / "run")


@pytest.mark.parametrize(
    "header,rows",
    [
        (["observation_id", "value", "value"], [["001", "1", "2"]]),
        (["observation_id", "other"], [["001", "1"]]),
        (["observation_id", "value", "undeclared"], [["001", "1", "x"]]),
        (["observation_id", " value"], [["001", "1"]]),
        (["observation_id", "value"], [["001"]]),
        (["observation_id", "value"], [["001", "1", "extra"]]),
        (["observation_id", "value"], [["001", "1"], ["001", "2"]]),
        (["observation_id", "value"], [["other", "1"]]),
        (["observation_id", "value"], []),
        (["observation_id", "value"], [[" 001", "1"]]),
        (["observation_id", "value"], [["", "1"]]),
        (["observation_id", "value"], [["001", "1"], []]),
    ],
)
def test_malformed_headers_rows_and_observation_links_fail(tmp_path, header, rows):
    _, config = make_source(tmp_path, rows, header=header)
    with pytest.raises(TabularIngestionError):
        ingest_tabular(config, base_dir=tmp_path, output_dir=tmp_path / "run")
    assert not (tmp_path / "run/summary.json").exists()


@pytest.mark.parametrize(
    "raw", [b"", b"observation_id,value\n001,\xff\n", b'observation_id,value\n001,"unfinished\n']
)
def test_empty_invalid_utf8_and_malformed_quotes_are_diagnostic(tmp_path, raw):
    path, config = make_source(tmp_path, [])
    path.write_bytes(raw)
    with pytest.raises(TabularIngestionError):
        ingest_tabular(config, base_dir=tmp_path, output_dir=tmp_path / "run")
    assert (tmp_path / "run/failure.json").exists()


def test_utf8_bom_crlf_quoted_delimiters_and_newlines_are_preserved(tmp_path):
    label = 'forest, "edge"\nsecond line'
    path, config = make_source(tmp_path, [["001", label]], dtype="categorical", categories=(label,))
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert b"\r\n" in path.read_bytes()
    assert ingest_tabular(config, base_dir=tmp_path, output_dir=tmp_path / "run").tables[
        0
    ].values == ((label,),)


def test_source_unit_column_is_checked_against_config(tmp_path):
    options = TabularReaderOptions(unit_id_column="unit")
    _, config = make_source(
        tmp_path,
        [["001", "1", "wrong"]],
        options=options,
        header=["observation_id", "value", "unit"],
    )
    with pytest.raises(TabularIngestionError, match="unit ID conflicts"):
        ingest_tabular(config, base_dir=tmp_path, output_dir=tmp_path / "run")


def test_all_missing_columns_and_analytical_rows_remain_visible(tmp_path):
    _, config = make_source(tmp_path, [["001", ""]])
    loaded = ingest_tabular(config, base_dir=tmp_path, output_dir=tmp_path / "run")
    assert loaded.summary["modalities"][0]["all_missing_columns"] == ["value"]
    assert loaded.summary["modalities"][0]["all_missing_analytical_observation_ids"] == ["001"]


def test_multitable_failure_retains_diagnostics_without_completion(source, tmp_path):
    raw, config = source
    with (raw / "environment.tsv").open("a") as handle:
        handle.write("undeclared\tforest\t1\t2\n")
    with pytest.raises(TabularIngestionError):
        ingest_tabular(config, base_dir=raw, output_dir=tmp_path / "run")
    assert (tmp_path / "run/modality_000/table.json").is_file()
    assert (tmp_path / "run/modality_001/input/environment.tsv").is_file()
    assert read_json(tmp_path / "run/failure.json")["status"] == "failed"
    assert not (tmp_path / "run/summary.json").exists()


def test_selection_preserves_config_order_and_does_not_open_unselected_sources(source, tmp_path):
    raw, config = source
    (raw / "environment.tsv").unlink()
    loaded = ingest_tabular(
        config, base_dir=raw, output_dir=tmp_path / "run", modality_names=["external_traits"]
    )
    assert len(loaded.tables) == 1 and len(loaded.summary["provenance"]["inputs"]) == 1


@pytest.mark.parametrize(
    "names", [[], "external_traits", ["missing"], ["external_traits"] * 2, [1]]
)
def test_invalid_selection_fails_before_output_creation(source, tmp_path, names):
    raw, config = source
    with pytest.raises(TabularIngestionError):
        ingest_tabular(config, base_dir=raw, output_dir=tmp_path / "run", modality_names=names)
    assert not (tmp_path / "run").exists()


def test_input_change_during_staging_is_detected(source, tmp_path, monkeypatch):
    from super_deli.io import staging

    raw, config = source
    copyfile = staging.shutil.copyfile

    def change_after_copy(source, destination):
        copyfile(source, destination)
        source.write_bytes(source.read_bytes() + b"\n")

    monkeypatch.setattr(staging.shutil, "copyfile", change_after_copy)
    with pytest.raises(OSError, match="changed while staging"):
        ingest_tabular(config, base_dir=raw, output_dir=tmp_path / "run")
    assert (tmp_path / "run/failure.json").exists()


def test_cli_json_selection_and_errors(source, tmp_path, capsys):
    raw, _ = source
    args = [
        "ingest-tabular",
        str(raw / "dataset.json"),
        "--modality",
        "site_environment",
        "--modality",
        "external_traits",
        "--output-dir",
        str(tmp_path / "cli"),
    ]
    assert main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert [m["name"] for m in summary["modalities"]] == ["external_traits", "site_environment"]
    assert main(args) == 2
    assert "File exists" in capsys.readouterr().err


def test_tabular_import_and_run_do_not_import_scientific_libraries(tmp_path):
    code = """
import sys
from pathlib import Path
from super_deli import load_config
from super_deli.io.tabular import ingest_tabular
source = Path(sys.argv[1])
ingest_tabular(load_config(source), base_dir=source.parent, output_dir=sys.argv[2])
assert not {'numpy', 'pandas', 'sklearn', 'snpio'} & sys.modules.keys()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(EXAMPLES / "tabular/dataset.json"),
            str(tmp_path / "subprocess"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
