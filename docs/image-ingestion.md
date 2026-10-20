# Image ingestion

This checkpoint adds local PDF, PNG, JPEG (`.jpg`/`.jpeg`) and HEIC/HEIF ingestion
on Python 3.11/3.12. It preserves original files and creates 8-bit RGB/RGBA PNG
display derivatives with explicit biological links and decoding provenance.
The separate [landmark workflow](landmark-detection.md) detects visible markers.
This adapter performs no TPS/image alignment, shape modeling or species inference.

```bash
python -m pip install -e '.[dev,images]'
super-deli ingest-images examples/images/dataset.json \
  --output-dir /tmp/super-deli-images
```

Choose a new output directory. All declared `image_manifest` modalities are read
by default; repeat `--modality NAME` to select a subset. Unselected source files
need not exist, although all dataset declarations must remain valid.

## Declaration and manifest

Add a morphometric modality to a version-one dataset with declared biological
units. This example is the modality object, not a complete dataset:

```json
{
  "name": "photographs",
  "kind": "morphometric",
  "level": "specimen",
  "format": "image_manifest",
  "path": "images/manifest.csv",
  "observations": [
    {"observation_id": "head_front", "unit_id": "specimen1"},
    {"observation_id": "head_side", "unit_id": "specimen1"}
  ],
  "image_options": {
    "observation_id_column": "observation_id",
    "pdf_dpi": 150,
    "max_pixels": 40000000,
    "max_source_bytes": 268435456,
    "heif_hdr_policy": "reject"
  }
}
```

Every option above is optional and shows its default. Options apply only to
`image_manifest`; omitting the whole object preserves previous configuration
hashes. `pdf_dpi` is an integer from 1 through 1200. Pixel and file-byte limits
must be positive integers. `heif_hdr_policy` accepts `reject` or `to_8bit`.
Unknown fields and ambiguous numeric types are errors.

```csv
observation_id,image_path,page,frame
head_front,phone.HEIC,,
head_side,scan.pdf,2,
```

The UTF-8 CSV requires the observation-ID column and `image_path`; a BOM is
accepted. Column order is free. Paths resolve relative to the **original
manifest's parent**, independently of the working directory. Absolute local
paths are accepted. URLs are not fetched. Extensions are case insensitive and
must agree with decoded content.

Observation IDs must match the declared set exactly. Duplicates, missing or
undeclared records, malformed rows and unknown columns fail. IDs are never
trimmed or matched by row position. Source row order is retained. Repeated views
or measurements use distinct observation IDs linked to the same biological unit.
Specimen, population and locality levels are supported; no links are inferred.
Optional `unit_id` must agree with the declaration. Optional `specimen_id` is
valid only for specimen-level records and must match their declared unit.

`page` selects **one-based PDF pages**; `frame` selects **zero-based raster/HEIF
image indices**. A source with multiple pages/frames requires an explicit
selector. A singleton defaults to page 1 or frame 0. A row describes exactly one
selected image. Use separate observation IDs to ingest several pages/frames of
the same source. Out-of-range, negative or inapplicable selectors fail. HEIF
indices refer to the images exposed by the decoder, not thumbnails/depth maps.
Animated PNG frames are decoded as Pillow's displayed/composited frames.

The reader also accepts the established standalone manifest columns:
`image_path`, `specimen_id`, `group`, `split`, `tps_path`, `landmark_count`,
`scale_value`, `scale_unit`, `check_status`, `check_details`.
For a legacy manifest, explicitly set
`image_options.observation_id_column="specimen_id"` and declare matching IDs.
Legacy IDs then identify both the observation and its specimen; repeated views
need the separate observation-ID schema shown above. Blank/`ok` check statuses
are accepted; other statuses fail. Group, split, TPS and scale values are retained
as raw metadata. They are never used as cluster labels, opened as TPS files or
applied to pixels. `image_tps_manifest.py` now recognizes the additional image
extensions; its CSV schema and metadata-only behavior remain unchanged.

## Display and measurement boundaries

PNG/JPEG use Pillow with EXIF transpose once, including mirrored orientations.
HEIF uses libheif's display transforms through the standalone pillow-heif API;
its Pillow conversion resets EXIF/XMP orientation, and the reader applies no
additional rotation. This avoids rotating an already displayed HEIF image twice.
See the primary [pillow-heif API](https://pillow-heif.readthedocs.io/en/latest/reference/API.html)
and [Pillow EXIF transpose documentation](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html#PIL.ImageOps.exif_transpose).
No global HEIF Pillow plugin is registered.

PDFium renders selected pages, including intrinsic rotation, annotations and
forms, against white. Rendering uses `pdf_dpi / 72` display scale; page size,
bounding box, rotation, nominal DPI and PDFium version are recorded. PDF canvas
units, cropping and any document-specific UserUnit do not establish biological
scale. See [PDFium's Python rendering API](https://pypdfium2.readthedocs.io/en/stable/python_api.html).
Password-protected PDFs have no password option in this checkpoint.

Raster dimensions are preserved apart from orientation; there is no arbitrary
resizing, segmentation, feature extraction or landmark localization. PNG
derivatives use top-left-origin raster coordinates. Existing TPS coordinates may
use another origin, axis direction, orientation or image frame. This reader
does not transform or reconcile them. The coordinate adapter continues to treat
its `IMAGE=` references as metadata only. Matching observation IDs alone does
not establish geometric correspondence.

Output PNGs preserve transparency and are lossless encodings of the **decoded
display pixels**. JPEG/HEIF source compression and PDF rendering can already be
lossy. Grayscale, palette and CMYK inputs are explicitly converted to RGB/RGBA;
the original mode and conversion are recorded. RGB/RGBA ICC profiles are retained
and hashed. Profiles for other input modes are hashed but omitted from converted
output because their original color space no longer applies. There is no ICC
color-space conversion or colorimetric calibration. HEIF NCLX metadata is
recorded. These are display derivatives, not calibrated color measurements.

16-bit PNG is rejected to prevent silent reduction. HEIF above 8 bits is rejected
by default. Explicit `heif_hdr_policy="to_8bit"` permits the decoder's conversion,
records source bit depth and reduction, and retains the high-bit-depth original.
That conversion is not a scientific intensity-preserving transformation or a
validated HDR tone-mapping workflow.

## Resource limits, artifacts and API

Each manifest and source must fit `max_source_bytes`; each selected raster must
fit `max_pixels`. Dimensions are checked before pixel loading/PDF rendering and
after raster frame selection. No input is downsampled to fit a budget. These
guards do not hard-bound native decoder memory, PDF intermediate resources,
container parsing, or total run size. PDFium calls made by this reader are
serialized; applications using PDFium elsewhere concurrently should isolate
decoding in separate processes. Pillow warning policy changes are scoped and its
global pixel threshold is not modified.

Each modality writes its staged manifest, verified original source copies,
numbered PNGs, and `images.json`. A source shared by several observations is
staged once per modality. Records include source/derivative relative paths and
SHA-256 hashes, observation/unit IDs, actual page/frame, dimensions, mode,
decoding decisions and manifest metadata. `summary.json` contains coverage,
effective options, selected input hashes and package/native-library versions.
Original files are never rewritten. A post-creation failure writes
`failure.json`, retains partial diagnostics and produces no top-level completion
summary. Declaration, missing dependency and manifest-file preflight failures
happen before creating the output directory. Existing outputs are never replaced.

```python
from pathlib import Path
from super_deli import load_config
from super_deli.io.images import ingest_images

source = Path("examples/images/dataset.json").absolute()
loaded = ingest_images(
    load_config(source),
    base_dir=source.parent,
    output_dir=Path("/tmp/super-deli-images-api"),
    modality_names=["photographs"],
)
records = loaded.collections[0].records
```

Root imports, configuration validation and the other ingestion adapters do not
require image libraries. Install `super-deli[images]` for this adapter. The tiny
synthetic fixtures in `examples/images/` cover every supported format, rotated
EXIF/HEIF, transparent PNG and selected/rotated PDF pages. `expected.json` states
dimensions, modes and independent corner-color truth with compression tolerance.
`generate_fixtures.py NEW_DIRECTORY` regenerates them with recorded codec
versions. They test ingestion, not biological landmark or species truth; this
checkpoint adds no empirical data downloads or inference models.
