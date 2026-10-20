"""Real image codec truth, resource policies, orientation, identity and source safety."""

import json
import shutil
import warnings
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("PIL.Image")
pillow_heif = pytest.importorskip("pillow_heif")
pytest.importorskip("pypdfium2")
from PIL import Image  # noqa: E402

from super_deli import ImageReaderOptions, Observation, load_config  # noqa: E402
from super_deli.cli import main  # noqa: E402
from super_deli.core.provenance import configuration_hash, fingerprint_file  # noqa: E402
from super_deli.io.config import read_json  # noqa: E402
from super_deli.io.image_decode import decode_image  # noqa: E402
from super_deli.io.images import ImageIngestionError, ingest_images  # noqa: E402

pytestmark = pytest.mark.images
EXAMPLES = Path(__file__).resolve().parents[1] / "examples/images"


def snapshot(directory):
    return {
        str(p.relative_to(directory)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in directory.rglob("*")
        if p.is_file()
    }


@pytest.fixture
def source(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    shutil.copytree(EXAMPLES, raw)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return raw, load_config(raw / "dataset.json")


def single(config, name, **options):
    modality = config.modalities[0]
    return replace(
        config,
        modalities=(
            replace(
                modality,
                observations=tuple(o for o in modality.observations if o.observation_id == name),
                image_options=replace(modality.image_options, **options),
            ),
        ),
    )


def manifest_one(raw, name, path, *, page="", frame=""):
    import csv

    with (raw / "manifest.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["observation_id", "image_path", "page", "frame"])
        writer.writerow([name, str(path), page, frame])


def test_real_codec_truth_links_provenance_and_source_safety(source, tmp_path):
    raw, config = source
    before = snapshot(raw)
    globals_before = Image.MAX_IMAGE_PIXELS, list(warnings.filters)
    loaded = ingest_images(config, base_dir=raw, output_dir=tmp_path / "run")
    truth = read_json(raw / "expected.json")["observations"]
    records = loaded.collections[0].records
    assert [r.observation_id for r in records] == ["png", "jpeg", "heic", "heif", "pdf1", "pdf2"]
    for record in records:
        expected = truth[record.observation_id]
        with Image.open(loaded.output_dir / record.image_ref) as image:
            assert list(image.size) == expected["size"]
            assert image.mode == expected["mode"]
            w, h = image.size
            points = [(5, 5), (w - 6, 5), (5, h - 6), (w - 6, h - 6)]
            for point, color in zip(points, expected["corners"], strict=True):
                assert all(
                    abs(a - b) <= expected["tolerance"]
                    for a, b in zip(image.getpixel(point)[:3], color, strict=True)
                )
            assert image.getexif().get(274, 1) == 1
            if record.observation_id == "png":
                assert image.getpixel((0, 0))[3] == 0
        assert (
            fingerprint_file(loaded.output_dir / record.source_ref).sha256 == record.source_sha256
        )
        assert fingerprint_file(loaded.output_dir / record.image_ref).sha256 == record.image_sha256
    assert records[-1].source_ref == records[-2].source_ref  # shared PDF staged only once
    assert {r.observation_id: r.unit_id for r in records} == {
        o.observation_id: o.unit_id for o in config.modalities[0].observations
    }
    summary = loaded.summary["modalities"][0]
    assert summary["source_count"] == 5 and summary["image_count"] == 6
    assert summary["coverage"]["missing_unit_ids"] == ("specimen4",)
    assert summary["coverage"]["repeated_observation_count"] == 3
    assert loaded.summary["provenance"]["configuration_sha256"] == configuration_hash(
        config.to_dict()
    )
    assert len(loaded.summary["provenance"]["inputs"]) == 6
    assert {"Pillow", "pillow-heif", "pypdfium2", "libheif", "pdfium"} <= loaded.summary[
        "provenance"
    ]["software"].keys()
    assert snapshot(raw) == before and not list(Path.cwd().iterdir())
    assert globals_before == (Image.MAX_IMAGE_PIXELS, warnings.filters)
    assert (loaded.output_dir / "summary.json").is_file()
    assert not (loaded.output_dir / "failure.json").exists()


@pytest.mark.parametrize(
    "orientation,indices",
    [
        (1, [0, 1, 2, 3]),
        (2, [1, 0, 3, 2]),
        (3, [3, 2, 1, 0]),
        (4, [2, 3, 0, 1]),
        (5, [0, 2, 1, 3]),
        (6, [2, 0, 3, 1]),
        (7, [3, 1, 2, 0]),
        (8, [1, 3, 0, 2]),
    ],
)
def test_all_exif_orientations_exactly_once(tmp_path, orientation, indices):
    path = tmp_path / "oriented.png"
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    with Image.new("RGB", (2, 2)) as original:
        original.putdata(colors)
        exif = Image.Exif()
        exif[274] = orientation
        original.save(path, exif=exif)
    decoded, *_, metadata = decode_image(path, page=None, frame=None, options=ImageReaderOptions())
    with decoded:
        assert list(decoded.getdata()) == [colors[i] for i in indices]
        assert decoded.getexif().get(274, 1) == 1
    assert metadata["exif_orientation"] == orientation


@pytest.mark.parametrize("kind", ["pdf", "apng", "heif"])
def test_multicontainer_requires_explicit_selection(tmp_path, kind):
    path = tmp_path / {"pdf": "pages.pdf", "apng": "animation.png", "heif": "sequence.heif"}[kind]
    if kind == "pdf":
        shutil.copyfile(EXAMPLES / "pages.pdf", path)
    else:
        with Image.new("RGB", (16, 12), "red") as red, Image.new("RGB", (16, 12), "blue") as blue:
            if kind == "apng":
                red.save(path, save_all=True, append_images=[blue], duration=100)
            else:
                container = pillow_heif.from_pillow(red)
                container.add_from_pillow(blue)
                container.save(path, save_all=True, quality=-1, chroma=444)
    options = ImageReaderOptions(pdf_dpi=72)
    with pytest.raises(ImageIngestionError, match="select .* explicitly"):
        decode_image(path, page=None, frame=None, options=options)
    page, frame = (2, None) if kind == "pdf" else (None, 1)
    decoded, *_ = decode_image(path, page=page, frame=frame, options=options)
    with decoded:
        r, g, b = decoded.getpixel((5, 5))
        assert r < 3 and g < 3 and b > 250
        if kind == "pdf":
            assert decoded.size == (24, 48)
    with pytest.raises(ImageIngestionError, match="out of range"):
        decode_image(
            path,
            page=9 if kind == "pdf" else None,
            frame=None if kind == "pdf" else 9,
            options=options,
        )


def test_pdf_resolution_and_preallocation_guard(tmp_path):
    path = EXAMPLES / "pages.pdf"
    decoded, *_, metadata = decode_image(
        path, page=1, frame=None, options=ImageReaderOptions(pdf_dpi=144)
    )
    with decoded:
        assert decoded.size == (144, 72)
    assert metadata["pdf_dpi"] == 144
    with pytest.raises(ImageIngestionError, match="max_pixels"):
        decode_image(path, page=1, frame=None, options=ImageReaderOptions(max_pixels=100))


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".heic", ".heif", ".pdf"])
def test_corrupt_sources_have_diagnostics_and_no_complete_marker(source, tmp_path, suffix):
    raw, config = source
    (raw / ("broken" + suffix)).write_bytes(b"corrupted source")
    manifest_one(raw, "png", "broken" + suffix, page="1" if suffix == ".pdf" else "")
    with pytest.raises(ImageIngestionError):
        ingest_images(single(config, "png"), base_dir=raw, output_dir=tmp_path / "run")
    assert read_json(tmp_path / "run/failure.json")["status"] == "failed"
    assert not (tmp_path / "run/summary.json").exists()


def test_extension_mismatch_and_invalid_orientation(tmp_path):
    path = tmp_path / "wrong.jpg"
    shutil.copyfile(EXAMPLES / "pattern.png", path)
    with pytest.raises(ImageIngestionError, match="disagree"):
        decode_image(path, page=None, frame=None, options=ImageReaderOptions())
    path = tmp_path / "bad.png"
    with Image.new("RGB", (2, 2)) as im:
        exif = Image.Exif()
        exif[274] = 9
        im.save(path, exif=exif)
    with pytest.raises(ImageIngestionError, match="orientation"):
        decode_image(path, page=None, frame=None, options=ImageReaderOptions())


def test_high_bit_depth_policy_is_explicit(tmp_path):
    png = tmp_path / "16bit.png"
    with Image.new("I;16", (16, 12), 32768) as im:
        im.save(png)
    with pytest.raises(ImageIngestionError, match="16-bit PNG"):
        decode_image(png, page=None, frame=None, options=ImageReaderOptions())
    heif = tmp_path / "10bit.heic"
    pillow_heif.from_bytes("RGB;16", (16, 12), b"\x00\x80" * (16 * 12 * 3)).save(
        heif, bit_depth=10, quality=-1, chroma=444
    )
    with pytest.raises(ImageIngestionError, match="explicit heif_hdr_policy"):
        decode_image(heif, page=None, frame=None, options=ImageReaderOptions())
    decoded, *_, metadata = decode_image(
        heif, page=None, frame=None, options=ImageReaderOptions(heif_hdr_policy="to_8bit")
    )
    with decoded:
        assert decoded.mode == "RGB"
    assert metadata["input_bit_depth"] == 10 and metadata["bit_depth_reduced"]


@pytest.mark.parametrize("mode", ["L", "LA", "P", "CMYK", "RGB"])
def test_explicit_color_conversion_and_icc_policy(tmp_path, mode):
    path = tmp_path / ("color.jpg" if mode == "CMYK" else "color.png")
    with Image.new(mode, (16, 12)) as im:
        im.save(path, icc_profile=b"fixture profile bytes")
    decoded, *_, metadata = decode_image(path, page=None, frame=None, options=ImageReaderOptions())
    with decoded:
        assert decoded.mode == ("RGBA" if mode == "LA" else "RGB")
        assert ("icc_profile" in decoded.info) == (mode == "RGB")
    assert metadata["input_mode"] == mode
    assert metadata["icc_profile_sha256"]
    assert metadata["icc_profile_preserved"] == (mode == "RGB")


@pytest.mark.parametrize("policy", ["pixels", "bytes", "manifest"])
def test_resource_limits_and_failure_boundary(source, tmp_path, policy):
    raw, config = source
    manifest_one(raw, "png", "pattern.png")
    options = {"max_pixels": 10} if policy == "pixels" else {"max_source_bytes": 100}
    if policy == "manifest":
        options["max_source_bytes"] = 1
    with pytest.raises(ImageIngestionError):
        ingest_images(single(config, "png", **options), base_dir=raw, output_dir=tmp_path / "run")
    if policy == "manifest":
        assert not (tmp_path / "run").exists()
    else:
        assert (tmp_path / "run/failure.json").exists()
        assert not (tmp_path / "run/summary.json").exists()


def test_nested_manifest_paths_output_safety_and_cli(source, tmp_path, capsys):
    raw, config = source
    subdir = raw / "images with spaces"
    subdir.mkdir()
    shutil.copyfile(raw / "pattern.png", subdir / "sample.PNG")
    manifest_one(subdir, "png", "sample.PNG")
    modality = replace(single(config, "png").modalities[0], path="images with spaces/manifest.csv")
    config = replace(config, modalities=(modality,))
    from super_deli import write_config

    write_config(config, raw / "nested.json")
    out = tmp_path / "run"
    assert (
        main(
            [
                "ingest-images",
                str(raw / "nested.json"),
                "--output-dir",
                str(out),
                "--modality",
                modality.name,
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    before = snapshot(out)
    with pytest.raises(FileExistsError):
        ingest_images(config, base_dir=raw, output_dir=out)
    assert snapshot(out) == before


@pytest.mark.parametrize("names", [[], "photographs", ["missing"], ["photographs"] * 2, [1]])
def test_invalid_selection_precedes_output(source, tmp_path, names):
    raw, config = source
    with pytest.raises(ImageIngestionError):
        ingest_images(config, base_dir=raw, output_dir=tmp_path / "run", modality_names=names)
    assert not (tmp_path / "run").exists()


def test_selected_modalities_and_partial_failure(source, tmp_path):
    raw, config = source
    missing = replace(config.modalities[0], name="unavailable", path="absent.csv")
    multiple = replace(config, modalities=(*config.modalities, missing))
    assert ingest_images(
        multiple, base_dir=raw, output_dir=tmp_path / "selected", modality_names=["photographs"]
    ).collections
    (raw / "absent.csv").write_text("observation_id,image_path\npng,broken.png\n")
    with pytest.raises(ImageIngestionError):
        ingest_images(multiple, base_dir=raw, output_dir=tmp_path / "failed")
    assert (tmp_path / "failed/modality_000/images.json").exists()
    assert (tmp_path / "failed/failure.json").exists()
    assert not (tmp_path / "failed/summary.json").exists()


@pytest.mark.parametrize("level", ["population", "locality"])
def test_nonindividual_links_are_supported(source, tmp_path, level):
    raw, config = source
    manifest_one(raw, "png", str((raw / "pattern.png").absolute()))
    modality = replace(
        single(config, "png").modalities[0], level=level, observations=(Observation("png", "unit"),)
    )
    config = replace(
        config,
        modalities=(modality,),
        populations=("unit",) if level == "population" else (),
        localities=("unit",) if level == "locality" else (),
    )
    (collection,) = ingest_images(config, base_dir=raw, output_dir=tmp_path / "run").collections
    assert collection.level == level and collection.records[0].unit_id == "unit"
