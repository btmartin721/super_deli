# TPS and fixed-landmark coordinates

This checkpoint reads fixed landmarks from TPS or long-form coordinate CSV into
typed, linked observations. It uses the Python standard library on Python
3.11/3.12. The separate [image adapter](image-ingestion.md) decodes image manifests.
The separate [marker workflow](landmark-detection.md) detects visible marks.
The separate [shape workflow](shape-pca.md) adds fixed-landmark GPA and PCA.
Anatomical learning, semilandmark sliding and new delimitation models are later work.

```bash
python -m pip install -e '.[dev]'
super-deli ingest-morphometric examples/morphometric/dataset.json \
  --output-dir /tmp/super-deli-landmark-example
```

All `morphometric` modalities with format `tps` or `landmarks_csv` are loaded by
default. Select one or more with repeated `--modality NAME` options. Collections
follow configuration order. Unselected sources are not opened; their declarations
must still be valid. Relative source paths resolve against the dataset JSON's
parent. Use a new output directory for each run.

## Declare geometry and correspondence

Each selected modality must have `morphometric_options`:

```json
{
  "dimensions": 2,
  "landmark_ids": ["nose", "left", "right"],
  "coordinate_unit": "pixel",
  "scale_unit": "mm",
  "tps_id_field": "ID",
  "missing_values": ["", "NA"],
  "missing_landmark": [-1, -1]
}
```

`dimensions` must be 2 or 3. `landmark_ids` is a required, ordered, nonempty list
of unique names. In TPS, coordinate row i receives declared landmark name i.
CSV names each landmark and is explicitly reordered to this declaration, with
the original per-observation landmark order also recorded. Equal names are the
user's assertion of correspondence; ingestion cannot verify biological homology.
No cross-modality alignment or correspondence is inferred.

Only dimensions and landmark IDs are required. Defaults are
`coordinate_unit=null`, `scale_unit=null`, `tps_id_field="ID"`,
`missing_values=[""]`, and `missing_landmark=[]` (no sentinel). A null unit means
unspecified. Options apply only to TPS/landmark CSV modalities. Old dataset
declarations can still be validated without these options, but coordinate
ingestion requires them. Omitting the new field preserves existing configuration
serialization and hashes.

## TPS subset

The common TPS conventions distinguish 2D `LM=` from 3D `LM3=` records and permit
ID/image identifiers and scale factors; see the primary
[geomorph TPS reader documentation](https://search.r-project.org/CRAN/refmans/geomorph/html/readland.tps.html).
Super Deli implements this explicit fixed-landmark subset:

```text
LM=3
0 0
4 0
0 3
ID=measurement_001
IMAGE=images/specimen1.HEIC
SCALE=0.5
COMMENT=Manual digitization
```

Multiple records per file are supported. `LM3=` requires three coordinate fields
on each line. Every block must have exactly the declared number of landmarks and
the declared dimensions. Blocks start with `LM=`/`LM3=`, followed by all coordinates,
then metadata. Blank lines are allowed. Directive keys are case-insensitive;
numeric fields can contain surrounding structural whitespace. Identifier values
are exact, case-sensitive, nonempty and unpadded.

The default observation identifier is the full `ID=` value. Explicitly set
`tps_id_field="IMAGE"` to use the full `IMAGE=` value instead. Paths/extensions
are not shortened, normalized or stripped. The selected field is required in
every record. There is no automatic row-number or filename fallback. Nonselected
ID/image metadata remains in the output. Source record order is preserved.

`ID=`, `IMAGE=` and `SCALE=` can occur at most once per block. Repeated `COMMENT=`
lines are preserved in order. Unknown directives, global metadata before the
first block, malformed/truncated blocks, extra coordinate lines, curves, sliders,
radii and other unsupported sections fail explicitly. Semilandmark definitions
must be handled in a later reader/model contract; they are not silently dropped
or treated as independent fixed landmarks.

## Coordinate CSV

Use one row per observation/landmark pair, with exact headers:

```csv
observation_id,landmark_id,x,y,unit_id,scale_factor,image
measurement_001,nose,0,0,specimen1,0.5,images/specimen1.png
measurement_001,left,4,0,specimen1,0.5,images/specimen1.png
measurement_001,right,0,3,specimen1,0.5,images/specimen1.png
```

Required columns are `observation_id`, `landmark_id`, `x`, `y`, plus `z` for 3D.
Optional columns are `unit_id`, `scale_factor` and `image`. Column order is free;
undeclared/duplicate/padded headers and wrong-width rows fail. A 2D source cannot
contain an extra z column. CSV uses commas with standard quoting, UTF-8 (optional
initial BOM), and LF or CRLF. Invalid UTF-8/quoting and blank records fail.

Rows can be interleaved across observations. Collections retain first-appearance
observation order. Every declared landmark must have exactly one row per source
observation, including rows whose coordinates are missing. Omitted rows are errors.
Unknown names and duplicate pairs fail. Wide coordinate tables are not inferred.
If `unit_id` is present, every row must match the observation declaration. Scale
and image metadata must be consistent across all rows of an observation,
including blank-versus-present values; the reader does not propagate one row's
metadata to blank entries. Empty optional scale/image fields become null.

## Missingness, units and biological links

Coordinates accept finite decimal/scientific notation. Overflow, nonzero
underflow to zero, nonfinite values and nonnumeric placeholders fail unless
explicitly declared coordinate missing tokens. Missing tokens are exact and
case-sensitive and become None/null, with a per-coordinate Boolean mask. The
reader retains partial-coordinate missingness as well as wholly absent points.
TPS fields are whitespace-separated, so use a nonblank token such as `NA` there.

Negative coordinates are valid. They are not automatically missing. A configured
`missing_landmark` sentinel must be a finite numeric vector matching the declared
dimensions; only an exact whole-vector match becomes a fully missing point.
For example, with `[-1, -1]` declared missing, `[-1, 2]` is still observed.
Numeric coordinate missing tokens can additionally mark individual axes when
explicitly configured. Source copies and options preserve what was interpreted.

`SCALE=` / `scale_factor` is optional metadata and must be positive and finite
when present. It is **never multiplied into the coordinates in this checkpoint**.
`coordinate_unit` describes the stored coordinates; `scale_unit` records the
intended target unit if the factor is later applied. An absent factor remains
null, not 1. Mixed available/missing calibration is reported. No unit, origin,
axis direction, image orientation, reflection or coordinate-frame conversion is
inferred. Images are metadata strings only: no existence check, file read, hash,
PDF page selection, or PNG/JPEG/HEIC/HEIF decoding is performed.

Observation IDs must exactly cover the modality's declarations. Missing,
undeclared or duplicate TPS observations fail; multiple CSV rows belong to one
observation only through their explicit IDs. Repeat observations can link to the
same biological unit and remain separate records. The declared specimen,
population or locality level is retained without aggregation or broadcasting.
Missing unit coverage, absent/partial landmarks and calibration gaps are reported.

All-missing or geometrically degenerate configurations can be ingested and must
be assessed before analysis. No centroid size, distances, Procrustes alignment,
missing-point estimation, shape PCA, uncertainty or significance is computed.
Parsing consistency does not establish independence, landmark quality or
statistical readiness.

## Python API and outputs

```python
from pathlib import Path

from super_deli import load_config
from super_deli.io.morphometric import ingest_morphometric

source = Path("examples/morphometric/dataset.json").absolute()
loaded = ingest_morphometric(
    load_config(source),
    base_dir=source.parent,
    output_dir="/tmp/new-landmark-api-run",
    modality_names=["outline"],
)
collection = loaded.collections[0]
record = collection.records[0]
print(collection.landmark_ids, record.coordinates, record.missing_mask)
```

The API returns `LoadedMorphometricData` with immutable tuple-based
`LandmarkCollection`/`LandmarkRecord` values, an output directory and summary.
No scientific library or image decoder is required. Each new directory contains
numbered `modality_000/input/<filename>` copies, `modality_000/landmarks.json`
typed geometry and a final `summary.json` completion marker. Geometry JSON
retains IDs, biological level, declared order, coordinates, masks, units,
calibration and TPS comments. Summaries include coverage, missingness, effective
options, configuration/software provenance and selected-source SHA-256 hashes.
Modality names are never output path components.

Inputs and existing outputs are preserved. Failures after creation leave
`failure.json` and staged/partial diagnostics without a completion summary.
CLI success returns 0; errors return 2. Source rows are parsed sequentially,
but records and JSON output are held in memory; out-of-core ingestion is not
implemented. The standalone `image_tps_manifest.py` retains its agreed manifest
columns and single-block metadata parser, separate from these coordinate readers.
Its image extension discovery now also includes PDF/HEIC/HEIF.

Stored fixtures provide hand-authored coordinate/identity truth for both formats,
including 2D/3D records, repeats and missingness. They do not provide empirical
species truth or validate downstream morphometric inference.
