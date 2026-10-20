"""Create tiny codec fixtures in a new directory with hand-specified display truth."""

import csv
from importlib.metadata import version
from pathlib import Path

import pillow_heif
from PIL import Image, ImageDraw

from super_deli.io.config import write_json


def _pdf(path):
    # Independent, minimal vector PDF: a quadrant page and a rotated blue page.
    streams = [
        b"1 0 0 rg 0 18 36 18 re f\n0 1 0 rg 36 18 36 18 re f\n"
        b"0 0 1 rg 0 0 36 18 re f\n1 1 0 rg 36 0 36 18 re f\n",
        b"0 0 1 rg 0 0 48 24 re f\n",
    ]
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 72 36] /Resources << >> /Contents 5 0 R >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 48 24] /Rotate 90 "
        b"/Resources << >> /Contents 6 0 R >>",
    ]
    objects += [
        b"<< /Length " + str(len(s)).encode() + b" >>\nstream\n" + s + b"endstream" for s in streams
    ]
    data, offsets = b"%PDF-1.4\n", [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data += str(index).encode() + b" 0 obj\n" + obj + b"\nendobj\n"
    start = len(data)
    data += b"xref\n0 7\n0000000000 65535 f \n" + b"".join(
        f"{i:010d} 00000 n \n".encode() for i in offsets[1:]
    )
    data += b"trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n" + str(start).encode() + b"\n%%EOF\n"
    path.write_bytes(data)


def generate(destination: Path):
    destination.mkdir(exist_ok=False)
    with Image.new("RGB", (64, 40)) as pattern:
        draw = ImageDraw.Draw(pattern)
        for box, color in [
            ((0, 0, 31, 19), "red"),
            ((32, 0, 63, 19), "lime"),
            ((0, 20, 31, 39), "blue"),
            ((32, 20, 63, 39), "yellow"),
        ]:
            draw.rectangle(box, fill=color)
        with pattern.convert("RGBA") as alpha:
            alpha.putpixel((0, 0), (255, 0, 0, 0))
            alpha.save(destination / "pattern.png")
        exif = Image.Exif()
        exif[274] = 6
        pattern.save(destination / "photo.jpg", quality=100, subsampling=0, exif=exif)
        # Supply raw pixels so the encoder applies the orientation only once.
        pillow_heif.from_bytes("RGB", pattern.size, pattern.tobytes()).save(
            destination / "phone.heic", quality=-1, chroma=444, exif=exif.tobytes()
        )
        pillow_heif.from_pillow(pattern).save(destination / "phone.heif", quality=-1, chroma=444)
    _pdf(destination / "pages.pdf")
    rows = [
        ("png", "specimen1", "pattern.png", "", ""),
        ("jpeg", "specimen1", "photo.jpg", "", ""),
        ("heic", "specimen2", "phone.heic", "", ""),
        ("heif", "specimen3", "phone.heif", "", ""),
        ("pdf1", "specimen2", "pages.pdf", "1", ""),
        ("pdf2", "specimen3", "pages.pdf", "2", ""),
    ]
    with (destination / "manifest.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["observation_id", "specimen_id", "image_path", "page", "frame"])
        writer.writerows(rows)
    write_json(
        {
            "schema_version": 1,
            "dataset_id": "illustrative_image_codecs",
            "specimens": [{"specimen_id": f"specimen{i}"} for i in range(1, 5)],
            "modalities": [
                {
                    "name": "photographs",
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
    normal = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]
    clockwise = [normal[i] for i in (2, 0, 3, 1)]
    expected = {
        "description": "Synthetic display truth; inset corner probes; no biological landmarks",
        "generating_software": {name: version(name) for name in ("Pillow", "pillow-heif")},
        "observations": {
            "png": {"size": [64, 40], "corners": normal, "tolerance": 0, "mode": "RGBA"},
            "jpeg": {"size": [40, 64], "corners": clockwise, "tolerance": 6, "mode": "RGB"},
            "heic": {"size": [40, 64], "corners": clockwise, "tolerance": 6, "mode": "RGB"},
            "heif": {"size": [64, 40], "corners": normal, "tolerance": 6, "mode": "RGB"},
            "pdf1": {"size": [72, 36], "corners": normal, "tolerance": 0, "mode": "RGB"},
            "pdf2": {"size": [24, 48], "corners": [[0, 0, 255]] * 4, "tolerance": 0, "mode": "RGB"},
        },
    }
    write_json(expected, destination / "expected.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="New fixture directory")
    generate(parser.parse_args().destination)
