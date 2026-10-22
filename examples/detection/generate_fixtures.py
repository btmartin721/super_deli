"""Generate synthetic visible-marker images and explicit display-coordinate truth."""

import csv
from importlib.metadata import version
from pathlib import Path

import pillow_heif
from PIL import Image, ImageDraw

from super_deli.io.config import write_json


def generate(destination: Path) -> None:
    destination.mkdir(exist_ok=False)
    normal = {"red_mark": [20, 20], "green_mark": [80, 20], "blue_mark": [50, 55]}
    rotated = {name: [79 - y, x] for name, (x, y) in normal.items()}
    with Image.new("RGB", (120, 80), (245, 245, 245)) as pattern:
        draw = ImageDraw.Draw(pattern)
        for (x, y), color in zip(normal.values(), ("red", "lime", "blue"), strict=True):
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        pattern.save(destination / "marked.png")
        exif = Image.Exif()
        exif[274] = 6
        pattern.save(destination / "marked.jpg", quality=95, subsampling=0, exif=exif)
        pillow_heif.from_bytes("RGB", pattern.size, pattern.tobytes()).save(
            destination / "marked.heic", quality=90, chroma=444, exif=exif.tobytes()
        )
        pillow_heif.from_pillow(pattern).save(destination / "marked.heif", quality=90, chroma=444)
        with pattern.transpose(Image.Transpose.ROTATE_270) as second:
            pattern.save(
                destination / "marked.pdf",
                format="PDF",
                resolution=72,
                save_all=True,
                append_images=[second],
            )
    with Image.new("RGB", (120, 80), (245, 245, 245)) as problem:
        draw = ImageDraw.Draw(problem)
        for x, y in ((20, 20), (45, 25)):
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="red")
        draw.point((100, 70), fill="red")
        draw.ellipse((-3, 52, 3, 58), fill="blue")
        problem.save(destination / "problem.png")
    rows = [
        ("png", "specimen1", "marked.png", ""),
        ("jpeg", "specimen1", "marked.jpg", ""),
        ("heic", "specimen2", "marked.heic", ""),
        ("heif", "specimen2", "marked.heif", ""),
        ("pdf1", "specimen1", "marked.pdf", "1"),
        ("pdf2", "specimen2", "marked.pdf", "2"),
        ("problem", "specimen2", "problem.png", ""),
    ]
    with (destination / "manifest.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("observation_id", "specimen_id", "image_path", "page"))
        writer.writerows(rows)
    write_json(
        {
            "schema_version": 1,
            "dataset_id": "synthetic_visible_markers",
            "specimens": [{"specimen_id": f"specimen{i}"} for i in (1, 2, 3)],
            "modalities": [
                {
                    "name": "marked_views",
                    "kind": "morphometric",
                    "level": "specimen",
                    "format": "image_manifest",
                    "path": "manifest.csv",
                    "image_options": {"pdf_dpi": 72},
                    "observations": [
                        {"observation_id": o, "unit_id": u} for o, u, *_ in reversed(rows)
                    ],
                }
            ],
        },
        destination / "dataset.json",
    )
    write_json(
        {
            "schema_version": 1,
            "method": "rgb_components",
            "landmarks": [
                {
                    "landmark_id": "red_mark",
                    "rgb_lower": [180, 0, 0],
                    "rgb_upper": [255, 90, 90],
                    "min_area": 9,
                    "max_area": 150,
                },
                {
                    "landmark_id": "green_mark",
                    "rgb_lower": [0, 180, 0],
                    "rgb_upper": [90, 255, 90],
                    "min_area": 9,
                    "max_area": 150,
                },
                {
                    "landmark_id": "blue_mark",
                    "rgb_lower": [0, 0, 180],
                    "rgb_upper": [90, 90, 255],
                    "min_area": 9,
                    "max_area": 150,
                },
            ],
        },
        destination / "detector.json",
    )
    write_json(
        {
            "description": "Synthetic displayed marker centers; no biological accuracy claim",
            "generating_software": {name: version(name) for name in ("Pillow", "pillow-heif")},
            "coordinate_tolerance_pixels": 0.75,
            "observations": {
                name: {
                    "coordinates": rotated if name in {"jpeg", "heic", "pdf2"} else normal,
                    "size": [80, 120] if name in {"jpeg", "heic", "pdf2"} else [120, 80],
                }
                for name in ("png", "jpeg", "heic", "heif", "pdf1", "pdf2")
            },
            "problem_status": {
                "red_mark": "ambiguous",
                "green_mark": "missing",
                "blue_mark": "boundary",
            },
        },
        destination / "expected.json",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="New fixture directory")
    generate(parser.parse_args().destination)
