"""Deterministic visible-marker centroids; no anatomical learning or confidence scores."""

from __future__ import annotations

import math
from importlib.metadata import PackageNotFoundError, version

from super_deli.core.detection import LandmarkDetection, MarkerCandidate
from super_deli.models.marker_config import MarkerDetectorConfig, MarkerRule


def detector_dependencies() -> dict[str, str]:
    try:
        import numpy  # noqa: F401
        from scipy import ndimage  # noqa: F401

        return {name: version(name) for name in ("numpy", "scipy")}
    except (ImportError, OSError, PackageNotFoundError) as exc:
        raise RuntimeError(
            "Landmark detection requires: pip install 'super-deli[images,landmarks]'"
        ) from exc


def _detect_rule(pixels, rule: MarkerRule, config: MarkerDetectorConfig) -> LandmarkDetection:
    import numpy as np
    from scipy import ndimage

    height, width = pixels.shape[:2]
    # Pixel centers use integer x/y; regions select x/width and y/height in
    # half-open intervals. Shared ROI edges therefore cannot claim one pixel twice.
    x0, y0, x1, y1 = (
        math.ceil(rule.search_box[i] * size)
        for i, size in enumerate((width, height, width, height))
    )
    box = (x0, y0, x1, y1)
    crop = pixels[y0:y1, x0:x1]
    mask = np.all((crop[:, :, :3] >= rule.rgb_lower) & (crop[:, :, :3] <= rule.rgb_upper), axis=2)
    mask &= crop[:, :, 3] >= config.min_alpha
    structure = ndimage.generate_binary_structure(2, 1 if config.connectivity == 4 else 2)
    labels, count = ndimage.label(mask, structure=structure)
    if count > config.max_components:
        raise ValueError(f"{rule.landmark_id}: component count {count} exceeds max_components")
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    eligible = np.flatnonzero((areas >= rule.min_area) & (areas <= rule.max_area))
    eligible = eligible[eligible != 0]
    candidates = []
    if len(eligible):
        boxes = ndimage.find_objects(labels, max_label=count)
        centers = ndimage.center_of_mass(mask, labels, eligible)
        for index, (cy, cx) in zip(eligible, centers, strict=True):
            rows, columns = boxes[index - 1]
            bounds = (x0 + columns.start, y0 + rows.start, x0 + columns.stop, y0 + rows.stop)
            boundary = bounds[0] == x0 or bounds[1] == y0 or bounds[2] == x1 or bounds[3] == y1
            candidates.append(
                MarkerCandidate(
                    int(areas[index]), (float(cx + x0), float(cy + y0)), bounds, boundary
                )
            )
    coordinate = (None, None)
    if count == 0:
        status, reason = "missing", "No pixels satisfy the declared color/alpha/search rules"
    elif not candidates:
        status, reason = "rejected", "All components fall outside the declared area range"
    elif len(candidates) > 1:
        status, reason = "ambiguous", "Multiple area-eligible components; identity is unresolved"
    elif candidates[0].touches_search_boundary:
        status, reason = "boundary", "Component touches the search/image edge; clipping is possible"
    else:
        status, reason = "detected", "Unique area-eligible interior component; provisional centroid"
        coordinate = candidates[0].centroid
    return LandmarkDetection(
        rule.landmark_id,
        status,
        coordinate,
        reason,
        box,
        int(count),
        int(count - len(candidates)),
        tuple(candidates),
    )


def detect_markers(image, *, config: MarkerDetectorConfig) -> tuple[LandmarkDetection, ...]:
    """Locate explicit visible markers in an already display-oriented Pillow image.

    Coordinates are x/y pixel centers, origin top-left, x right and y down.
    Missing/ambiguous/clipped points stay null. Component geometry is not a
    confidence interval and a unique centroid does not establish anatomical truth.
    """
    import numpy as np

    config = MarkerDetectorConfig.from_dict(config.to_dict())
    width, height = image.size
    if width < 1 or height < 1 or width * height > config.max_pixels:
        raise ValueError(f"Detection dimensions {image.size} exceed max_pixels={config.max_pixels}")
    if image.mode not in {"RGB", "RGBA"}:
        raise ValueError("Detection requires an RGB/RGBA display derivative from image ingestion")
    with image.convert("RGBA") as rgba:
        pixels = np.array(rgba, dtype=np.uint8)
    return tuple(_detect_rule(pixels, rule, config) for rule in config.landmarks)
