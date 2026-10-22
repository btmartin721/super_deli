"""Known marker positions, conservative unresolved outputs and end-to-end artifacts."""

import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy.ndimage")
pytest.importorskip("PIL.Image")
pytest.importorskip("pillow_heif")
pytest.importorskip("pypdfium2")
from PIL import Image, ImageDraw  # noqa: E402

from super_deli import load_config  # noqa: E402
from super_deli.cli import main  # noqa: E402
from super_deli.core.provenance import configuration_hash, fingerprint_file  # noqa: E402
from super_deli.io.config import read_json  # noqa: E402
from super_deli.io.morphometric import ingest_morphometric  # noqa: E402
from super_deli.models.marker_config import MarkerDetectorConfig, MarkerRule  # noqa: E402
from super_deli.models.marker_detection import detect_markers  # noqa: E402
from super_deli.workflows import landmark_detection as workflow  # noqa: E402

pytestmark = pytest.mark.landmarks
EXAMPLES = Path(__file__).resolve().parents[1] / "examples/detection"


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
    return (
        raw,
        load_config(raw / "dataset.json"),
        MarkerDetectorConfig.from_dict(read_json(raw / "detector.json")),
    )


def red_rule(**kwargs):
    return MarkerRule(
        "red", (200, 0, 0), (255, 50, 50), **{"min_area": 1, "max_area": 100, **kwargs}
    )


def one(image, rule=None, **kwargs):
    return detect_markers(
        image, config=MarkerDetectorConfig(landmarks=(rule or red_rule(),), **kwargs)
    )[0]


def test_centroid_is_unweighted_pixel_geometry_and_input_unchanged():
    with Image.new("RGB", (10, 10), "white") as image:
        image.putpixel((3, 4), (200, 0, 0))
        image.putpixel((4, 4), (255, 0, 0))
        before = image.tobytes()
        result = one(image, red_rule(min_area=2, max_area=2))
        assert result.status == "detected" and result.coordinate == (3.5, 4.0)
        assert result.candidates[0].bounding_box == (3, 4, 5, 5)
        assert image.tobytes() == before


def test_ambiguous_components_are_not_ranked_by_size_or_position():
    with Image.new("RGB", (30, 20), "white") as image:
        draw = ImageDraw.Draw(image)
        draw.rectangle((2, 2, 4, 4), fill="red")
        draw.rectangle((15, 10, 19, 14), fill="red")
        result = one(image)
        assert result.status == "ambiguous" and result.coordinate == (None, None)
        assert len(result.candidates) == 2


@pytest.mark.parametrize("fill,status", [("white", "missing"), ("red", "rejected")])
def test_missing_or_oversized_components_stay_null(fill, status):
    with Image.new("RGB", (20, 20), fill) as image:
        result = one(image, red_rule(max_area=10))
        assert result.status == status and result.coordinate == (None, None)


def test_search_regions_preserve_named_correspondence_and_rounding():
    rule = red_rule()
    config = MarkerDetectorConfig(
        landmarks=(
            replace(rule, landmark_id="left", search_box=(0, 0, 0.5, 1)),
            replace(rule, landmark_id="right", search_box=(0.5, 0, 1, 1)),
        )
    )
    with Image.new("RGB", (11, 10), "white") as image:
        image.putpixel((3, 4), (255, 0, 0))
        image.putpixel((8, 4), (255, 0, 0))
        left, right = detect_markers(image, config=config)
        assert left.coordinate == (3.0, 4.0) and right.coordinate == (8.0, 4.0)
        assert left.search_box_pixels[2] == right.search_box_pixels[0] == 6
    # A normalized interval narrower than one pixel can select no pixel centers.
    with Image.new("RGB", (10, 10), "red") as image:
        empty = one(image, red_rule(search_box=(0.101, 0.101, 0.102, 0.102)))
        assert empty.status == "missing"


@pytest.mark.parametrize(
    "point,box",
    [
        ((0, 5), (0, 0, 1, 1)),
        ((9, 5), (0, 0, 1, 1)),
        ((5, 0), (0, 0, 1, 1)),
        ((5, 9), (0, 0, 1, 1)),
        ((3, 5), (0.3, 0, 0.8, 1)),
    ],
)
def test_image_and_search_edge_clipping_are_unresolved(point, box):
    with Image.new("RGB", (10, 10), "white") as image:
        image.putpixel(point, (255, 0, 0))
        result = one(image, red_rule(search_box=box))
        assert result.status == "boundary" and result.coordinate == (None, None)


def test_connectivity_is_explicit_and_alpha_threshold_is_inclusive():
    with Image.new("RGBA", (10, 10), (255, 255, 255, 255)) as image:
        image.putpixel((3, 3), (255, 0, 0, 255))
        image.putpixel((4, 4), (255, 0, 0, 255))
        image.putpixel((7, 7), (255, 0, 0, 0))
        assert one(image, connectivity=4).status == "ambiguous"
        assert one(image, connectivity=8).coordinate == (3.5, 3.5)
        image.putpixel((4, 4), (255, 0, 0, 254))
        assert one(image).coordinate == (3.0, 3.0)
        assert one(image, min_alpha=254).coordinate == (3.5, 3.5)


def test_computational_limits_and_unsupported_pixel_modes():
    with Image.new("RGB", (20, 20), "white") as image:
        with pytest.raises(ValueError, match="max_pixels"):
            one(image, max_pixels=10)
        for x in range(2, 18, 2):
            image.putpixel((x, 5), (255, 0, 0))
        with pytest.raises(ValueError, match="max_components"):
            one(image, max_components=3)
    with Image.new("L", (10, 10)) as image, pytest.raises(ValueError, match="RGB/RGBA"):
        one(image)


def test_all_format_truth_exports_overlays_and_provenance(source, tmp_path):
    raw, dataset, config = source
    before = snapshot(raw)
    run = workflow.run_landmark_detection(
        dataset, detector_config=config, base_dir=raw, output_dir=tmp_path / "run"
    )
    truth = read_json(raw / "expected.json")
    diagnostics = read_json(run.output_dir / "detections.json")["records"]
    assert [r.observation_id for r in run.collection.records] == [
        "png",
        "jpeg",
        "heic",
        "heif",
        "pdf1",
        "pdf2",
        "problem",
    ]
    errors = []
    for record, diagnostic in zip(run.collection.records, diagnostics, strict=True):
        assert diagnostic["unit_id"] == record.unit_id
        if record.observation_id in truth["observations"]:
            expected = truth["observations"][record.observation_id]
            assert [diagnostic["width"], diagnostic["height"]] == expected["size"]
            for name, point in zip(run.collection.landmark_ids, record.coordinates, strict=True):
                error = math.dist(point, expected["coordinates"][name])
                errors.append(error)
                assert error <= truth["coordinate_tolerance_pixels"]
            assert all(d["status"] == "detected" for d in diagnostic["landmarks"])
        else:
            assert {d["landmark_id"]: d["status"] for d in diagnostic["landmarks"]} == truth[
                "problem_status"
            ]
            assert all(all(p) for p in record.missing_mask)
        image_path = run.output_dir / diagnostic["image_ref"]
        overlay_path = run.output_dir / diagnostic["overlay_ref"]
        assert fingerprint_file(image_path).sha256 == diagnostic["image_sha256"]
        assert fingerprint_file(overlay_path).sha256 == diagnostic["overlay_sha256"]
        with Image.open(image_path) as image, Image.open(overlay_path) as overlay:
            assert image.size == overlay.size
            assert image.convert("RGBA").tobytes() != overlay.tobytes()
    assert len(errors) == 18
    assert (
        run.summary["unresolved_count"] == 3 and run.summary["detection_counts"]["detected"] == 18
    )
    assert run.summary["review_status"] == "unreviewed"
    assert run.summary["localization_uncertainty"]["status"] == "not_computed"
    provenance = run.summary["provenance"]
    assert provenance["configuration_sha256"] == configuration_hash(provenance["configuration"])
    assert {"numpy", "scipy", "libheif", "pdfium"} <= provenance["software"].keys()
    exported = load_config(run.output_dir / "landmark_dataset.json")
    roundtrip = ingest_morphometric(
        exported, base_dir=run.output_dir, output_dir=tmp_path / "coordinate_roundtrip"
    ).collections[0]
    assert [(r.observation_id, r.unit_id, r.coordinates, r.image) for r in roundtrip.records] == [
        (r.observation_id, r.unit_id, r.coordinates, r.image) for r in run.collection.records
    ]
    assert snapshot(raw) == before and not list(Path.cwd().iterdir())
    assert (run.output_dir / "summary.json").exists()
    assert not (run.output_dir / "failure.json").exists()


def test_selection_and_output_preservation(source, tmp_path):
    raw, dataset, config = source
    other = replace(dataset.modalities[0], name="absent", path="absent.csv")
    dataset = replace(dataset, modalities=(*dataset.modalities, other))
    for name in (None, "unknown"):
        with pytest.raises(ValueError, match="exactly one"):
            workflow.run_landmark_detection(
                dataset,
                detector_config=config,
                base_dir=raw,
                output_dir=tmp_path / "invalid",
                modality_name=name,
            )
        assert not (tmp_path / "invalid").exists()
    run = workflow.run_landmark_detection(
        dataset,
        detector_config=config,
        base_dir=raw,
        output_dir=tmp_path / "selected",
        modality_name="marked_views",
    )
    before = snapshot(run.output_dir)
    with pytest.raises(FileExistsError):
        workflow.run_landmark_detection(
            dataset,
            detector_config=config,
            base_dir=raw,
            output_dir=run.output_dir,
            modality_name="marked_views",
        )
    assert snapshot(run.output_dir) == before


def test_resource_failure_has_no_top_level_completion_marker(source, tmp_path):
    raw, dataset, config = source
    with pytest.raises(ValueError, match="max_pixels"):
        workflow.run_landmark_detection(
            dataset,
            detector_config=replace(config, max_pixels=10),
            base_dir=raw,
            output_dir=tmp_path / "run",
        )
    assert (tmp_path / "run/ingestion/summary.json").exists()
    assert (tmp_path / "run/failure.json").exists()
    assert not (tmp_path / "run/summary.json").exists()


def test_changed_derivative_is_detected(source, tmp_path, monkeypatch):
    raw, dataset, config = source
    original = workflow.detect_markers

    def mutate(image, *, config):
        result = original(image, config=config)
        with Path(image.filename).open("ab") as handle:
            handle.write(b"changed")
        return result

    monkeypatch.setattr(workflow, "detect_markers", mutate)
    with pytest.raises(OSError, match="changed during"):
        workflow.run_landmark_detection(
            dataset, detector_config=config, base_dir=raw, output_dir=tmp_path / "run"
        )
    assert (tmp_path / "run/failure.json").exists()
    assert not (tmp_path / "run/summary.json").exists()


def test_cli_complete_with_unresolved_points_and_invalid_config(source, tmp_path, capsys):
    raw, _, _ = source
    args = [
        "detect-landmarks",
        str(raw / "dataset.json"),
        "--detector-config",
        str(raw / "detector.json"),
        "--output-dir",
        str(tmp_path / "run"),
    ]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["unresolved_count"] == 3
    (raw / "detector.json").write_text('{"schema_version": 1}')
    assert main(args[:-1] + [str(tmp_path / "bad")]) == 2
    assert not (tmp_path / "bad").exists()
    assert capsys.readouterr().err.startswith("super-deli:")
