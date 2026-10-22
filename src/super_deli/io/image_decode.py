"""Scoped image/PDF decoders; no global Pillow plugin registration or source writes."""

from __future__ import annotations

import hashlib
import math
import warnings
from contextlib import closing
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import RLock

from super_deli.core.data import ImageReaderOptions
from super_deli.io.image_manifest import ImageIngestionError

_DECODE_LOCK = RLock()


def image_dependencies() -> dict:
    try:
        import pillow_heif
        import pypdfium2

        import_module("PIL.Image")
        software = {name: version(name) for name in ("Pillow", "pillow-heif", "pypdfium2")}
        software.update(
            libheif=pillow_heif.libheif_info()["libheif"], pdfium=str(pypdfium2.PDFIUM_INFO)
        )
        return software
    except (PackageNotFoundError, ImportError, OSError) as exc:
        raise ImageIngestionError(
            "Image ingestion requires: pip install 'super-deli[images]'"
        ) from exc


def _limit(size: tuple[int, int], options: ImageReaderOptions) -> None:
    if min(size) < 1 or size[0] * size[1] > options.max_pixels:
        raise ImageIngestionError(
            f"Decoded dimensions {size} exceed max_pixels={options.max_pixels}"
        )


def _choose(index: int | None, count: int, name: str, first: int) -> int:
    if index is None:
        if count != 1:
            raise ImageIngestionError(
                f"Source has {count} {name}s; select {name} explicitly in the manifest"
            )
        return first
    if not first <= index < first + count:
        raise ImageIngestionError(f"Selected {name}={index} is out of range for {count} {name}s")
    return index


def _rgb(image, metadata: dict):
    """Produce an 8-bit display derivative; all conversion choices remain explicit."""
    if image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "CMYK"}:
        raise ImageIngestionError(
            f"Unsupported decoded pixel mode {image.mode!r}; no implicit high-bit-depth cast"
        )
    mode = image.mode
    profile = image.info.get("icc_profile")
    keep_profile = bool(profile) and mode in {"RGB", "RGBA"}
    metadata.update(
        input_mode=mode,
        color_conversion=f"Pillow {mode} to RGB/RGBA; no ICC color-space conversion",
        icc_profile_sha256=hashlib.sha256(profile).hexdigest() if profile else None,
        icc_profile_preserved=keep_profile,
    )
    alpha = "A" in image.getbands() or "transparency" in image.info
    result = image.convert("RGBA" if alpha else "RGB")
    result.info.clear()  # no stale EXIF orientation in the normalized derivative
    if keep_profile:
        result.info["icc_profile"] = profile
    return result


def _raster(path, frame, options):
    from PIL import Image, ImageOps

    expected = "PNG" if path.suffix.lower() == ".png" else "JPEG"
    with Image.open(path, formats=("PNG", "JPEG")) as source:
        if source.format != expected:
            raise ImageIngestionError("Image extension and decoded content format disagree")
        count = getattr(source, "n_frames", 1)
        selected = _choose(frame, count, "frame", 0)
        _limit(source.size, options)
        source.seek(selected)
        _limit(source.size, options)
        bits = 8
        if expected == "PNG":
            with path.open("rb") as handle:
                header = handle.read(25)
            bits = header[24]
            if bits == 16:
                raise ImageIngestionError("16-bit PNG is not reduced to 8-bit implicitly")
        orientation = source.getexif().get(274, 1)
        if type(orientation) is not int or orientation not in range(1, 9):
            raise ImageIngestionError("Invalid EXIF orientation tag")
        metadata = {
            "frame_count": count,
            "input_dimensions": source.size,
            "input_bit_depth": bits,
            "exif_orientation": orientation,
            "orientation_handling": "Pillow EXIF transpose exactly once",
        }
        source.load()
        with ImageOps.exif_transpose(source) as displayed:
            output = _rgb(displayed, metadata)
    return output, expected, None, selected, metadata


def _heif(path, frame, options):
    import pillow_heif

    mime = pillow_heif.get_file_mimetype(path)
    if mime not in {"image/heic", "image/heif", "image/heic-sequence", "image/heif-sequence"}:
        raise ImageIngestionError("HEIC/HEIF extension does not contain supported HEIF content")
    container = pillow_heif.open_heif(path, convert_hdr_to_8bit=False)
    count = len(container)
    selected = _choose(frame, count, "frame", 0)
    source = container[selected]
    _limit(source.size, options)
    bits = source.info["bit_depth"]
    reduced = bits > 8
    if reduced:
        if options.heif_hdr_policy == "reject":
            raise ImageIngestionError(
                "High-bit-depth HEIF requires explicit heif_hdr_policy='to_8bit'"
            )
        container = pillow_heif.open_heif(path, convert_hdr_to_8bit=True)
        source = container[selected]
    with source.to_pillow() as displayed:
        metadata = {
            "mime_type": mime,
            "frame_count": count,
            "primary_frame_index": container.primary_index,
            "input_dimensions": source.size,
            "input_bit_depth": bits,
            "bit_depth_reduced": reduced,
            "nclx_profile": displayed.info.get("nclx_profile"),
            "original_orientation_metadata": displayed.info.get("original_orientation"),
            "orientation_handling": "libheif display transforms; EXIF/XMP reset; no extra rotation",
        }
        output = _rgb(displayed, metadata)
    return output, "HEIF", None, selected, metadata


def _pdf(path, page_number, options):
    import pypdfium2 as pdfium

    with pdfium.PdfDocument(path) as document:
        document.init_forms()
        selected = _choose(page_number, len(document), "page", 1)
        with closing(document.get_page(selected - 1)) as page:
            size = page.get_size()
            scale = options.pdf_dpi / 72
            _limit(tuple(math.ceil(d * scale) for d in size), options)
            metadata = {
                "page_count": len(document),
                "page_size_canvas_units": size,
                "page_rotation_degrees": page.get_rotation(),
                "page_bbox": page.get_bbox(),
                "pdf_dpi": options.pdf_dpi,
                "canvas_units_per_nominal_inch": 72,
                "pdfium_version": str(pdfium.PDFIUM_INFO),
                "orientation_handling": "PDFium displayed page including intrinsic rotation",
                "annotations_and_forms": "rendered",
                "background": "opaque white",
            }
            with closing(
                page.render(
                    scale=scale,
                    rev_byteorder=True,
                    draw_annots=True,
                    may_draw_forms=True,
                    fill_color=(255, 255, 255, 255),
                )
            ) as bitmap:
                with bitmap.to_pil() as displayed:
                    output = _rgb(displayed, metadata)
    _limit(output.size, options)
    return output, "PDF", selected, None, metadata


def decode_image(path: Path, *, page: int | None, frame: int | None, options: ImageReaderOptions):
    """Return an independent Pillow raster and explicit decoding metadata.

    Caller owns/closes the returned image. PDFium calls and local warning scopes
    are serialized; external concurrent PDFium users should use separate processes.
    """
    from PIL import Image

    try:
        with _DECODE_LOCK, warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                return _pdf(path, page, options)
            if suffix in {".heic", ".heif"}:
                return _heif(path, frame, options)
            return _raster(path, frame, options)
    except (
        ValueError,
        OSError,
        RuntimeError,
        SyntaxError,
        EOFError,
        Warning,
        Image.DecompressionBombError,
    ) as exc:
        if isinstance(exc, ImageIngestionError):
            raise
        raise ImageIngestionError(f"Could not decode {path.name}: {exc}") from exc
