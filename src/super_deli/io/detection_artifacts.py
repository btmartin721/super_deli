"""New coordinate exports and visual diagnostics for unreviewed marker detections."""

from __future__ import annotations

from pathlib import Path

from super_deli.core.data import DatasetConfig, Modality
from super_deli.core.detection import LandmarkDetection
from super_deli.core.morphometrics import LandmarkCollection
from super_deli.io.landmark_export import write_landmark_coordinates


def write_detection_overlay(image, detections: tuple[LandmarkDetection, ...], path: Path) -> None:
    """Draw provisional centroids and candidate boxes on a separate PNG copy.

    Numbers refer to landmark order in the detector config/diagnostics; using
    numeric labels also keeps rendering independent of identifier fonts/scripts.
    """
    from PIL import ImageDraw

    with image.convert("RGBA") as overlay:
        draw = ImageDraw.Draw(overlay)
        for index, detection in enumerate(detections, 1):
            color = (0, 180, 0, 255) if detection.status == "detected" else (255, 120, 0, 255)
            for candidate in detection.candidates:
                x0, y0, x1, y1 = candidate.bounding_box
                draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=1)
                x, y = candidate.centroid
                draw.text((x1 + 2, y0), str(index), fill=color)
                if detection.status == "detected":
                    draw.line((x - 4, y, x + 4, y), fill=color, width=1)
                    draw.line((x, y - 4, x, y + 4), fill=color, width=1)
        with path.open("xb") as handle:
            overlay.save(handle, format="PNG")


def write_detected_coordinates(
    collection: LandmarkCollection, *, dataset: DatasetConfig, source: Modality, directory: Path
) -> None:
    """Export all named rows, including null pairs, with an ingestible declaration."""
    write_landmark_coordinates(collection, dataset=dataset, source=source, directory=directory)
