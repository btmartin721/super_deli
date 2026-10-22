"""Dependency-free image declarations, manifest identity and optional import boundaries."""

import csv
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from super_deli import ImageReaderOptions, Observation, load_config, validate_dataset, write_config
from super_deli.core.schema import ConfigError, config_from_dict
from super_deli.io import images
from super_deli.io.image_manifest import (
    LEGACY_COLUMNS,
    ImageIngestionError,
    read_image_manifest,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def payload():
    return load_config(EXAMPLES / "images/dataset.json").to_dict()


def test_roundtrip_and_defaults(tmp_path):
    document = payload()
    document["modalities"][0]["image_options"] = {}
    config = config_from_dict(document)
    assert config.modalities[0].image_options == ImageReaderOptions()
    write_config(config, tmp_path / "config.json")
    assert load_config(tmp_path / "config.json") == config
    old = load_config(EXAMPLES / "morphometric/dataset.json").to_dict()
    assert all("image_options" not in m for m in old["modalities"])


@pytest.mark.parametrize(
    "options",
    [
        [],
        {"unknown": 1},
        {"pdf_dpi": True},
        {"pdf_dpi": 72.0},
        {"pdf_dpi": "72"},
        {"max_pixels": False},
        {"max_source_bytes": -1},
        {"max_source_bytes": 1.5},
        {"heif_hdr_policy": None},
        {"observation_id_column": ""},
    ],
)
def test_strict_option_types(options):
    document = payload()
    document["modalities"][0]["image_options"] = options
    with pytest.raises(ConfigError):
        config_from_dict(document)


@pytest.mark.parametrize(
    "options",
    [
        {"pdf_dpi": 0},
        {"pdf_dpi": 1201},
        {"max_pixels": 0},
        {"max_source_bytes": 0},
        {"heif_hdr_policy": "auto"},
        {"observation_id_column": "image_path"},
        {"observation_id_column": "frame"},
        {"observation_id_column": "page"},
    ],
)
def test_invalid_policies(options):
    document = payload()
    document["modalities"][0]["image_options"] = options
    assert not validate_dataset(
        config_from_dict(document), base_dir=EXAMPLES / "images", check_files=False
    ).valid


def test_options_are_scoped_to_image_manifests():
    document = payload()
    document["modalities"][0]["format"] = "tps"
    report = validate_dataset(
        config_from_dict(document), base_dir=EXAMPLES / "images", check_files=False
    )
    assert any(i.code == "invalid_image_options" for i in report.issues)


def manifest(tmp_path, rows, columns=("observation_id", "image_path")):
    path = tmp_path / "images.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    modality = replace(
        config_from_dict(payload()).modalities[0],
        observations=(Observation("001", "specimen1"),),
    )
    return path, modality


@pytest.mark.parametrize(
    "rows,columns",
    [
        ([], ("observation_id", "image_path")),
        ([("001", "a.png"), ("001", "a.png")], ("observation_id", "image_path")),
        ([("1", "a.png")], ("observation_id", "image_path")),
        ([(" 001", "a.png")], ("observation_id", "image_path")),
        ([("001", "a.svg")], ("observation_id", "image_path")),
        ([("001", "a.png", "2")], ("observation_id", "image_path", "page")),
        ([("001", "a.pdf", "0")], ("observation_id", "image_path", "page")),
        ([("001", "a.pdf", "0")], ("observation_id", "image_path", "frame")),
        ([("001", "a.png", "-1")], ("observation_id", "image_path", "frame")),
        ([("001", "a.png", "1.0")], ("observation_id", "image_path", "frame")),
        ([("001", "a.png", "wrong")], ("observation_id", "image_path", "unit_id")),
        ([("001", "a.png", "wrong")], ("observation_id", "image_path", "specimen_id")),
        ([("001", "a.png", "error")], ("observation_id", "image_path", "check_status")),
        ([("001", "a.png", "extra")], ("observation_id", "image_path")),
        ([("001", "a.png", "typo")], ("observation_id", "image_path", "unknown")),
        ([("001", "a.png", "a.png")], ("observation_id", "image_path", "image_path")),
    ],
)
def test_invalid_manifest_identity_shape_and_selectors(tmp_path, rows, columns):
    path, modality = manifest(tmp_path, rows, columns)
    with pytest.raises(ImageIngestionError):
        read_image_manifest(path, modality, source_dir=tmp_path)


def test_legacy_contract_and_metadata_are_preserved(tmp_path):
    # Compare to the authoritative standalone schema without requiring its import.
    import runpy

    legacy = runpy.run_path(str(EXAMPLES.parent / "image_tps_manifest.py"))
    assert LEGACY_COLUMNS == set(legacy["MANIFEST_COLUMNS"])
    columns = sorted(LEGACY_COLUMNS)
    row = dict.fromkeys(columns, "")
    row.update(
        specimen_id="specimen1",
        image_path="folder with spaces/photo.HEIC",
        tps_path="absent.tps",
        group="withheld truth",
        split="test",
        check_status="ok",
    )
    path, modality = manifest(tmp_path, [[row[c] for c in columns]], columns)
    modality = replace(
        modality,
        observations=(Observation("specimen1", "specimen1"),),
        image_options=ImageReaderOptions(observation_id_column="specimen_id"),
    )
    (ref,) = read_image_manifest(path, modality, source_dir=tmp_path)
    assert ref.path == tmp_path / row["image_path"]
    assert dict(ref.metadata)["group"] == "withheld truth"
    assert dict(ref.metadata)["tps_path"] == "absent.tps"


def test_bom_and_invalid_utf8(tmp_path):
    path, modality = manifest(tmp_path, [("001", "a.png")])
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert read_image_manifest(path, modality, source_dir=tmp_path)
    path.write_bytes(b"\xff")
    with pytest.raises(ImageIngestionError):
        read_image_manifest(path, modality, source_dir=tmp_path)


def test_missing_dependencies_fail_before_output(tmp_path, monkeypatch):
    def absent():
        raise ImageIngestionError("Install super-deli[images]")

    monkeypatch.setattr(images, "image_dependencies", absent)
    with pytest.raises(ImageIngestionError, match="super-deli"):
        images.ingest_images(
            config_from_dict(payload()), base_dir=EXAMPLES / "images", output_dir=tmp_path / "run"
        )
    assert not (tmp_path / "run").exists()


def test_core_and_configuration_import_no_codecs():
    code = """
import sys
from super_deli import load_config
from super_deli.io.images import ingest_images
load_config(sys.argv[1])
assert not {'PIL', 'pillow_heif', 'pypdfium2', 'snpio', 'numpy'} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(EXAMPLES / "images/dataset.json")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
