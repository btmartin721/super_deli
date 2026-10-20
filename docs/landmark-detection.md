# Visible-marker landmark detection

This checkpoint locates **visible, explicitly named markers** in images on
Python 3.11/3.12. Each landmark has a declared RGB color range, component-size
range and optional search region. This is a deterministic marker detector.
Anatomical landmark prediction on unmarked specimens needs annotated training
data and independent evaluation in a later checkpoint.

```bash
python -m pip install -e '.[dev,images,landmarks]'
super-deli detect-landmarks examples/detection/dataset.json \
  --detector-config examples/detection/detector.json \
  --output-dir /tmp/super-deli-detected-landmarks
```

The dataset uses the existing [image-manifest contract](image-ingestion.md).
Select exactly one `image_manifest` modality with `--modality NAME` when several
are declared. Original inputs and existing outputs are preserved. PDF pages,
raster frames, orientation, display conversion and source hashes are handled by
the image adapter. Detection sees those same PNG display derivatives without
another rotation or resize. Repeated observations and specimen/population/locality
links remain explicit; this workflow performs no biological aggregation.

## Named marker rules

The detector configuration is separate from dataset declarations. It is strict
JSON; unknown fields, missing required fields, incorrect types and duplicate
landmark IDs fail before processing. A complete one-marker configuration:

```json
{
  "schema_version": 1,
  "method": "rgb_components",
  "connectivity": 8,
  "min_alpha": 255,
  "max_pixels": 4000000,
  "max_components": 10000,
  "landmarks": [
    {
      "landmark_id": "anterior_marker",
      "rgb_lower": [180, 0, 0],
      "rgb_upper": [255, 90, 90],
      "min_area": 9,
      "max_area": 150,
      "search_box": [0, 0, 1, 1]
    }
  ]
}
```

`schema_version`, `method` and the `landmarks` array are required. Global fields
above show defaults. `connectivity` accepts 4 (edge neighbors) or 8 (including
diagonals). `min_alpha` is an inclusive integer threshold from 1 through 255;
fully transparent RGB values never qualify. Pixel/component budgets must be
positive integers. At most 128 named landmarks are supported per invocation.

Each rule requires `landmark_id`, `rgb_lower`, `rgb_upper`, `min_area` and
`max_area`. RGB bounds are inclusive integers from 0 through 255, applied to
decoded display channels without intensity weighting. Area bounds are inclusive
positive pixel counts; the upper bound must be at least the lower bound.
`search_box` defaults to the full image and uses normalized
`[left, top, right, bottom]` coordinates in `[0,1]` with positive dimensions.
It selects pixel centers satisfying `left <= x/width < right` and
`top <= y/height < bottom`; bounds therefore use ceilings when rasterized.
A tiny box can contain no pixel centers. Boxes and RGB ranges must be disjoint
in their combined color/space criteria: one pixel cannot qualify for two names.
Overlapping identity rules are rejected instead of assigning a blob arbitrarily.

Use different colors for different named landmarks, or disjoint regions for
markers sharing a color. Black/gray markers can use narrow RGB ranges and
separate search regions. The method does not read printed landmark numbers,
recognize biological anatomy, or recover TPS coordinates from an image.
Search regions refer to the **displayed** frame after EXIF/HEIF/PDF orientation.
One configuration applies to every selected observation; prepare consistent
views or use separate modalities/runs when protocols differ.

## Detection and unresolved points

Within each rule's region, the detector thresholds color and alpha, labels
connected components, and filters by component area. It uses SciPy's documented
[component labeling](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html)
and [center-of-mass](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.center_of_mass.html)
operations. Each surviving component's center is the unweighted mean of its
selected pixel centers; brightness is not mass. Output coordinates are `(x,y)`
in pixels with a top-left origin, x increasing right and y increasing down.
The first pixel center is `(0,0)`. Bounding boxes have exclusive right/bottom edges.

| Status | Meaning | Coordinate |
| --- | --- | --- |
| `detected` | Exactly one area-eligible component, wholly inside the search/image boundary | Provisional centroid |
| `missing` | No qualifying color/alpha pixels | Null pair |
| `rejected` | Components exist but all fall outside the area bounds | Null pair |
| `ambiguous` | Multiple area-eligible components | Null pair |
| `boundary` | The unique eligible component touches a search/image edge and may be clipped | Null pair |

All eligible candidates are retained in diagnostics, including their centroid,
pixel count, box and boundary flag. The detector records the total component
count and area-rejected count. It never selects the largest blob, sorts blobs
into anatomical names, fills a missing point or converts ambiguity to zero.
`status="complete"` and CLI exit 0 mean the workflow finished; unresolved markers
can remain. Inspect `unresolved_count`, point statuses and overlays before use.

## Review, uncertainty and scientific limits

Every output is `unreviewed`. An ink centroid is a marker location, not evidence
that the correct anatomical structure was marked. Background objects, text,
uneven lighting, compression, occlusion and merged markers can satisfy or change
the rules. A unique merged component can pass the filters. Review overlays and
diagnostics against the source images before using these coordinates.

Choose thresholds, spatial regions and areas using the imaging/marking protocol.
Pixel areas depend on image resolution and PDF DPI. Detection does not rescale
images to meet a rule or budget. RGB ranges operate on display pixels; image
ingestion supplies no colorimetric calibration. Existing TPS coordinates may
use different axes, origins or frames and are not automatically reconciled.
No physical scale, Procrustes alignment or species-delimitation model is applied.

Statistical localization confidence, confidence intervals and significance are
**not computed**: there is no independent annotated-image calibration in this
checkpoint. Component area, pixel spread, bounding boxes and the binary
`detected` status are not confidence estimates. A future anatomical model needs
paired named annotations, training/validation/test separation at the biological
unit level, localization-error evaluation and calibrated uncertainty. Rule
tuning should likewise use data separate from its final evaluation.

## Artifacts and API

A new run directory contains:

- `ingestion/`: verified source copies, display PNGs and image-ingestion provenance.
- `landmarks.json`: the existing 2D `LandmarkCollection` contract, preserving nulls and biological links.
- `landmarks.csv` and `landmark_dataset.json`: long-form coordinates and a matching configuration accepted by `ingest-morphometric`.
- `detections.json`: point statuses, reasons, candidates, image links/hashes and overlay hashes.
- `overlays/`: separate PNG copies with candidate boxes and provisional centroid crosses. Numeric labels refer to detector rule order; colors distinguish detected (green) and unresolved candidates (orange). Missing markers have no box.
- `summary.json`: completion, resolved/unresolved counts, frame conventions, unavailable uncertainty, exact dataset/detector configuration hash and software/native-codec provenance.

Coordinate CSV image references remain relative to the detection run root;
coordinate ingestion retains them as metadata. Keep the run artifacts together.
Display derivatives are hashed before and after detection. Source images are
never annotated in place. A failure after directory creation retains partial
artifacts and `failure.json`, with no **top-level** completion summary. A nested
image-ingestion summary can still exist if decoding finished before detection
failed. Existing output directories are never overwritten.

The detector's default four-million-pixel budget applies before numerical array
allocation and is separate from the image adapter's decoding budget. Set
`image_options.max_pixels` too if a lower decode limit is needed. The component
budget applies per image/rule before candidate enumeration. Rules are processed
sequentially, but arrays, candidate diagnostics and output records remain in
memory. These limits do not hard-bound native decoder memory or total run size.

```python
from pathlib import Path
from super_deli import load_config
from super_deli.io.config import read_json
from super_deli.models.marker_config import MarkerDetectorConfig
from super_deli.workflows.landmark_detection import run_landmark_detection

source = Path("examples/detection/dataset.json").absolute()
loaded = run_landmark_detection(
    load_config(source),
    detector_config=MarkerDetectorConfig.from_dict(read_json(source.parent / "detector.json")),
    base_dir=source.parent,
    output_dir=Path("/tmp/super-deli-landmark-api"),
)
coordinates = loaded.collection.records[0].coordinates
```

`landmarks` supplies NumPy/SciPy; install it together with `images` for this
workflow. Root imports and rule/configuration parsing remain dependency-free.
Synthetic fixtures cover all supported image formats, rotated JPEG/HEIC,
selected PDF pages and deliberately unresolved markers. Known centers are
specified independently of detection; tests check Euclidean error against a
declared 0.75-pixel tolerance across codecs. These are software/coordinate checks,
not empirical anatomical accuracy or uncertainty calibration. The complete
fixture generator records codec versions and writes only a new directory.
