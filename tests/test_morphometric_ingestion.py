"""TPS/CSV coordinate truth, correspondence, source safety and format boundaries."""

import csv
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from super_deli import Observation, load_config
from super_deli.cli import main
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.io.config import read_json
from super_deli.io.morphometric import MorphometricIngestionError, ingest_morphometric

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
    shutil.copytree(EXAMPLES / "morphometric", raw)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return raw, load_config(raw / "dataset.json")


def test_typed_truth_calibration_correspondence_and_original_preservation(source, tmp_path):
    raw, config = source
    before = snapshot(raw)
    loaded = ingest_morphometric(config, base_dir=raw, output_dir=tmp_path / "run")
    expected = read_json(raw / "expected.json")
    for collection, summary in zip(loaded.collections, loaded.summary["modalities"], strict=True):
        truth = expected[collection.modality_name]
        saved = read_json(loaded.output_dir / summary["artifact"])
        assert [r.observation_id for r in collection.records] == truth["observation_ids"]
        assert [r.unit_id for r in collection.records] == truth["unit_ids"]
        assert saved["landmark_ids"] == truth["landmark_ids"]
        assert [r["coordinates"] for r in saved["records"]] == truth["coordinates"]
        assert [r["missing_mask"] for r in saved["records"]] == truth["missing_mask"]
        assert [r.scale_factor for r in collection.records] == truth["scale_factors"]
    outline, surface = loaded.collections
    assert outline.coordinate_unit == "pixel" and outline.scale_unit == "mm"
    assert outline.records[0].coordinates[1] == (4.0, 0.0)  # factor 0.5 is metadata, not applied
    assert outline.records[-1].coordinates[0] == (-1.0, 2.0)  # legitimate negative coordinate
    assert outline.records[1].coordinates[1] == (None, None)  # explicit whole-point sentinel
    assert outline.records[0].image == "images/head3.HEIC"  # no image exists or is decoded
    assert outline.records[0].comments
    assert surface.records[0].source_landmark_order == ("tip", "base", "ventral", "dorsal")
    summaries = loaded.summary["modalities"]
    assert summaries[0]["coverage"]["repeated_observation_count"] == 1
    assert summaries[0]["coverage"]["missing_unit_ids"] == ("specimen2", "specimen4")
    assert summaries[0]["partial_landmarks_by_observation"]["outline1_repeat"] == 1
    assert summaries[0]["missing_landmarks_by_observation"]["outline1_repeat"] == 1
    assert summaries[0]["missing_scale_observation_ids"] == ["outline1_repeat"]
    assert loaded.summary["provenance"]["configuration_sha256"] == configuration_hash(
        config.to_dict()
    )
    for m, fingerprint in zip(
        config.modalities, loaded.summary["provenance"]["inputs"], strict=True
    ):
        assert fingerprint["sha256"] == fingerprint_file(raw / m.path).sha256
    assert snapshot(raw) == before and not list(Path.cwd().iterdir())
    assert not (loaded.output_dir / "failure.json").exists()
    assert not (loaded.output_dir / "result.json").exists()


def single_source(tmp_path, *, format="tps", dimensions=2, first="0", level="specimen"):
    points = [[first, "1", "2"][:dimensions], ["3", "4", "5"][:dimensions]]
    path = tmp_path / ("landmarks.tps" if format == "tps" else "landmarks.csv")
    if format == "tps":
        lm = "LM" if dimensions == 2 else "LM3"
        path.write_text(
            f"{lm}=2\n"
            + "\n".join(" ".join(p) for p in points)
            + "\nID=001\nIMAGE=folder/scale_LM=image.HEIC\nSCALE=0.5\n"
        )
    else:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["observation_id", "landmark_id", *("x", "y", "z")[:dimensions]])
            writer.writerows(
                [["001", name, *p] for name, p in zip(("a", "b"), points, strict=True)]
            )
    config = config_from_dict(
        {
            "schema_version": 1,
            "dataset_id": "geometry",
            "specimens": [{"specimen_id": "unit1"}] if level == "specimen" else [],
            "populations": ["unit1"] if level == "population" else [],
            "localities": ["unit1"] if level == "locality" else [],
            "modalities": [
                {
                    "name": "shape",
                    "kind": "morphometric",
                    "level": level,
                    "format": format,
                    "path": path.name,
                    "observations": [{"observation_id": "001", "unit_id": "unit1"}],
                    "morphometric_options": {
                        "dimensions": dimensions,
                        "landmark_ids": ["a", "b"],
                        "missing_values": ["", "NA"],
                    },
                }
            ],
        }
    )
    return path, config


@pytest.mark.parametrize("format", ["tps", "landmarks_csv"])
@pytest.mark.parametrize("dimensions", [2, 3])
@pytest.mark.parametrize("level", ["specimen", "population", "locality"])
def test_supported_dimensions_and_declared_biological_levels(tmp_path, format, dimensions, level):
    _, config = single_source(tmp_path, format=format, dimensions=dimensions, level=level)
    collection = ingest_morphometric(
        config, base_dir=tmp_path, output_dir=tmp_path / "run"
    ).collections[0]
    assert collection.dimensions == dimensions and collection.level == level
    assert len(collection.records[0].coordinates[0]) == dimensions
    assert collection.records[0].observation_id == "001"


@pytest.mark.parametrize("format", ["tps", "landmarks_csv"])
@pytest.mark.parametrize("first", ["nan", "Inf", "1e400", "1e-400", "1_000", "?", "12mm"])
def test_invalid_coordinate_values_fail_with_diagnostics(tmp_path, format, first):
    _, config = single_source(tmp_path, format=format, first=first)
    with pytest.raises(MorphometricIngestionError):
        ingest_morphometric(config, base_dir=tmp_path, output_dir=tmp_path / "run")
    assert read_json(tmp_path / "run/failure.json")["status"] == "failed"
    assert not (tmp_path / "run/summary.json").exists()


@pytest.mark.parametrize(
    "text",
    [
        "",
        "ID=001\n",
        "LM=2\n0 1\nID=001\n",
        "LM=3\n0 1\n3 4\nID=001\n",
        "LM=2\n0 1 2\n3 4\nID=001\n",
        "LM3=2\n0 1 2\n3 4 5\nID=001\n",
        "LM=2\n0 1\n3 4\n",
        "LM=2\n0 1\n3 4\nID=001\nID=001\n",
        "LM=2\n0 1\n3 4\nID=001\n5 6\n",
        "LM=2\n0 1\n3 4\nID=001\nCURVES=1\n",
        "LM=2\n0 1\n3 4\nID=001\nVARIABLES=1\n",
        "LM=2\n0 1\n3 4\nID= 001\n",
        "LM=2\n0 1\n3 4\nID=001\nSCALE=1\nSCALE=1\n",
    ],
)
def test_malformed_or_unsupported_tps_is_never_silently_dropped(tmp_path, text):
    path, config = single_source(tmp_path)
    path.write_text(text)
    with pytest.raises(MorphometricIngestionError):
        ingest_morphometric(config, base_dir=tmp_path, output_dir=tmp_path / "run")


@pytest.mark.parametrize("scale", ["0", "-1", "NaN", "1e400", "1e-400", "NA"])
def test_invalid_tps_scale_factors(tmp_path, scale):
    path, config = single_source(tmp_path)
    path.write_text(path.read_text().replace("SCALE=0.5", f"SCALE={scale}"))
    with pytest.raises(MorphometricIngestionError):
        ingest_morphometric(config, base_dir=tmp_path, output_dir=tmp_path / "run")


def test_tps_image_identity_is_explicit_and_preserves_full_path(tmp_path):
    path, config = single_source(tmp_path)
    modality = config.modalities[0]
    image_id = "folder/scale_LM=image.HEIC"
    config = replace(
        config,
        modalities=(
            replace(
                modality,
                observations=(Observation(image_id, "unit1"),),
                morphometric_options=replace(modality.morphometric_options, tps_id_field="IMAGE"),
            ),
        ),
    )
    path.write_text(path.read_text().replace("ID=001\n", ""))
    record = (
        ingest_morphometric(config, base_dir=tmp_path, output_dir=tmp_path / "run")
        .collections[0]
        .records[0]
    )
    assert record.observation_id == image_id and record.tps_id is None


def test_duplicate_and_missing_tps_observations_are_errors(tmp_path):
    path, config = single_source(tmp_path)
    path.write_text(path.read_text() * 2)
    with pytest.raises(MorphometricIngestionError, match="duplicate observation"):
        ingest_morphometric(config, base_dir=tmp_path, output_dir=tmp_path / "duplicate")
    path.write_text("LM=2\n0 1\n3 4\nID=undeclared\n")
    with pytest.raises(MorphometricIngestionError, match="undeclared observation"):
        ingest_morphometric(config, base_dir=tmp_path, output_dir=tmp_path / "unknown")
    path.write_text("LM=2\n0 1\n3 4\nID=001\n")
    modality = config.modalities[0]
    config = replace(
        config,
        modalities=(
            replace(
                modality, observations=(*modality.observations, Observation("absent", "unit1"))
            ),
        ),
    )
    with pytest.raises(MorphometricIngestionError, match="missing declared"):
        ingest_morphometric(config, base_dir=tmp_path, output_dir=tmp_path / "missing")


@pytest.mark.parametrize(
    "raw",
    [
        "observation_id,landmark_id,x,y\n001,a,0,1\n001,a,2,3\n",
        "observation_id,landmark_id,x,y\n001,a,0,1\n",
        "observation_id,landmark_id,x,y\n001,c,0,1\n001,b,2,3\n",
        "observation_id,landmark_id,x,y,z\n001,a,0,1,2\n001,b,3,4,5\n",
        "observation_id,landmark_id,x,y,unit_id\n001,a,0,1,wrong\n001,b,2,3,wrong\n",
        "observation_id,landmark_id,x,y,scale_factor\n001,a,0,1,1\n001,b,2,3,2\n",
        "observation_id,landmark_id,x,y,image\n001,a,0,1,a.png\n001,b,2,3,\n",
        "observation_id,landmark_id,x,y\n001,a,0,1\nother,b,2,3\n",
        "observation_id,landmark_id,x,y\n001,a,0\n001,b,2,3\n",
    ],
)
def test_csv_requires_complete_named_landmarks_and_consistent_metadata(tmp_path, raw):
    path, config = single_source(tmp_path, format="landmarks_csv")
    path.write_text(raw)
    with pytest.raises(MorphometricIngestionError):
        ingest_morphometric(config, base_dir=tmp_path, output_dir=tmp_path / "run")


@pytest.mark.parametrize("format", ["tps", "landmarks_csv"])
def test_bom_crlf_and_invalid_utf8(tmp_path, format):
    path, config = single_source(tmp_path, format=format)
    path.write_bytes(b"\xef\xbb\xbf" + path.read_text().replace("\n", "\r\n").encode())
    assert ingest_morphometric(config, base_dir=tmp_path, output_dir=tmp_path / "valid").collections
    path.write_bytes(b"\xff")
    with pytest.raises(MorphometricIngestionError):
        ingest_morphometric(config, base_dir=tmp_path, output_dir=tmp_path / "bad")


def test_fully_missing_geometry_is_retained_without_imputation(tmp_path):
    path, config = single_source(tmp_path)
    path.write_text("LM=2\nNA NA\nNA NA\nID=001\n")
    loaded = ingest_morphometric(config, base_dir=tmp_path, output_dir=tmp_path / "run")
    assert loaded.summary["modalities"][0]["all_missing_observation_ids"] == ["001"]
    assert loaded.collections[0].records[0].coordinates == ((None, None), (None, None))


def test_missing_geometry_options_and_existing_output_fail_safely(source, tmp_path):
    raw, config = source
    modality = replace(config.modalities[0], morphometric_options=None)
    with pytest.raises(MorphometricIngestionError, match="declare morphometric_options"):
        ingest_morphometric(
            replace(config, modalities=(modality,)), base_dir=raw, output_dir=tmp_path / "absent"
        )
    assert not (tmp_path / "absent").exists()
    output = tmp_path / "existing"
    output.mkdir()
    (output / "sentinel").write_text("keep")
    before = snapshot(output)
    with pytest.raises(FileExistsError):
        ingest_morphometric(config, base_dir=raw, output_dir=output)
    assert snapshot(output) == before


def test_selection_cli_and_multisource_failure_are_explicit(source, tmp_path, capsys):
    raw, config = source
    args = [
        "ingest-morphometric",
        str(raw / "dataset.json"),
        "--modality",
        "surface",
        "--modality",
        "outline",
        "--output-dir",
        str(tmp_path / "cli"),
    ]
    assert main(args) == 0
    assert [m["name"] for m in json.loads(capsys.readouterr().out)["modalities"]] == [
        "outline",
        "surface",
    ]
    (raw / "surface.csv").write_text("wrong,header\n")
    with pytest.raises(MorphometricIngestionError):
        ingest_morphometric(config, base_dir=raw, output_dir=tmp_path / "failed")
    assert (tmp_path / "failed/modality_000/landmarks.json").exists()
    assert (tmp_path / "failed/failure.json").exists()
    assert not (tmp_path / "failed/summary.json").exists()
    (raw / "surface.csv").unlink()
    assert ingest_morphometric(
        config, base_dir=raw, output_dir=tmp_path / "selected", modality_names=["outline"]
    ).collections


@pytest.mark.parametrize("names", [[], "outline", ["missing"], ["outline", "outline"], [1]])
def test_invalid_selection_fails_before_writing(source, tmp_path, names):
    raw, config = source
    with pytest.raises(MorphometricIngestionError):
        ingest_morphometric(config, base_dir=raw, output_dir=tmp_path / "run", modality_names=names)
    assert not (tmp_path / "run").exists()


def test_image_manifests_are_not_coordinate_sources(source, tmp_path):
    raw, config = source
    modality = replace(config.modalities[0], format="image_manifest", morphometric_options=None)
    with pytest.raises(MorphometricIngestionError, match="image manifests"):
        ingest_morphometric(
            replace(config, modalities=(modality,)), base_dir=raw, output_dir=tmp_path / "run"
        )
    assert not (tmp_path / "run").exists()


def test_morphometric_run_has_no_scientific_or_image_decoder_imports(tmp_path):
    code = """
import sys
from pathlib import Path
from super_deli import load_config
from super_deli.io.morphometric import ingest_morphometric
source = Path(sys.argv[1])
ingest_morphometric(load_config(source), base_dir=source.parent, output_dir=sys.argv[2])
assert not {'numpy', 'pandas', 'sklearn', 'snpio', 'PIL'} & sys.modules.keys()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(EXAMPLES / "morphometric/dataset.json"),
            str(tmp_path / "subprocess"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
