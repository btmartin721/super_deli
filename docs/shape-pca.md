# Fixed-landmark GPA and shape PCA

Python 3.11/3.12 supports fixed-landmark generalized Procrustes alignment (GPA),
orthogonal tangent-space projection and covariance PCA. This checkpoint provides
descriptive shape coordinates and scores. It adds no species-delimitation model,
group significance test, semilandmark sliding or calibrated uncertainty.

```bash
python -m pip install -e '.[dev,shape]'
super-deli shape-pca examples/shape/train_dataset.json \
  --model-config examples/shape/model.json --output-dir /tmp/super-deli-shape-fit
super-deli shape-pca examples/shape/projection_dataset.json \
  --model-config examples/shape/model.json --reference-run /tmp/super-deli-shape-fit \
  --output-dir /tmp/super-deli-shape-projection
```

Choose new output directories. The existing TPS/long-form coordinate CSV contract
supplies dimensions, ordered landmark IDs, observation links and missingness.
Use `--modality NAME` to select exactly one coordinate modality if several are
declared. Marker-detection exports can also supply this contract; review their
provisional coordinates before shape analysis. Matching names does not establish
anatomical homology or correct digitization.

## Explicit policies

A complete configuration showing defaults:

```json
{
  "schema_version": 1,
  "landmark_type": "fixed",
  "reflection": "forbid",
  "missing_policy": "error",
  "repeated_policy": "error",
  "n_components": null,
  "max_iter": 500,
  "tol": 1e-10,
  "rank_tol": 1e-12,
  "max_working_bytes": 268435456
}
```

Only `schema_version` is required; unknown fields and incorrect types fail.
`reflection="forbid"` uses proper rotations. Explicit `allow` permits reflections
and can remove chirality; it is not an automatic coordinate-axis correction.
Coordinate origins/axes and left/right landmark identities must follow a
consistent protocol before alignment. No landmark relabeling is inferred.

`missing_policy="error"` rejects incomplete observations. Explicit
`drop_observations` excludes whole observations with any missing coordinate and
records every exclusion. Landmarks are never dropped, interpolated or imputed.
At least two complete observations are needed for fitting, and one for projection.

`repeated_policy="error"` rejects repeated biological units among retained
observations. Explicit `allow` gives each observation equal weight, including
repeats; there is no aggregation or independence claim. Biological level and
unit IDs are preserved. Do not pool different digitizers of the same specimen
as independent specimens without a measurement-error/weighting protocol.

All points are fixed and equally weighted. At least three homologous landmarks
are required in 2D/3D. Planar 3D configurations are supported; collinear 3D
configurations cannot identify rotations. Zero centroid size, nonfinite values,
non-identifiable cross-covariance rotations and nonunique reflection-restricted
solutions fail explicitly. A 2D collinear configuration can still identify an
SO(2) rotation; it is not automatically rejected.

## Geometry and PCA

For each source configuration, the reader preserves raw coordinates and SCALE
metadata. Analysis subtracts the landmark centroid and divides by centroid size,
the square root of summed squared distances from that centroid. Normalization
uses an intermediate amplitude scale to avoid premature overflow/underflow.
Raw centroid positions/sizes are retained in the source coordinate units.
TPS SCALE is **not applied**, and physical units are never inferred. Uniform
scaling cancels in shape normalization; raw size values are not calibrated body
sizes. Removing uniform size does not remove allometric shape variation.

GPA alternates least-squares rotation to the current reference and normalization
of the equal-weight mean configuration. The first retained training observation
initializes the otherwise arbitrary global orientation. The convergence history
records mean-shape changes and squared residuals; the final configuration must
be stationary within `tol`. Exceeding `max_iter` fails with diagnostics. Numerical
convergence is not a guarantee of the global optimum for disparate shapes.
The method follows the standard fixed-point GPA framework described by
[geomorph's GPA documentation](https://rdrr.io/cran/geomorph/man/gpagen.html).

For aligned unit configuration `Y` and unit consensus `M`, the orthogonal tangent
residual is `T = Y - <Y,M>M`, using the Frobenius inner product. PCA centers the
flattened residuals and performs an economy SVD. Columns are in declared landmark
order, then x/y/z; coordinates are not standardized or whitened. Variances use
the sample denominator `n-1`. This is ordinary covariance PCA in a local tangent
plane, consistent with the standard ordination role described by
[geomorph's PCA documentation](https://rdrr.io/cran/geomorph/man/gm.prcomp.html).
This Python implementation does not depend on geomorph or R.

`rank_tol` defines numerical rank: singular values must exceed
`rank_tol * max(largest_singular_value, 1)`. Rank is bounded by `n-1` and the
fixed-shape dimension `k*d - d - 1 - d*(d-1)/2`. `n_components=null` retains all
resolved axes; a requested count is capped at rank. Numerically identical shapes
produce an explicit `no_resolved_shape_variation` status and zero PCA columns.
The aligned shapes remain available. PC signs are canonicalized by the largest
absolute loading, but tied eigenspaces have no unique individual axis basis.

Tangent projection requires the open consensus hemisphere. Diagnostics report
each observation's consensus angle and orthographic radial shrinkage
`1 - sin(angle)/angle`, with the zero-angle limit handled explicitly. These
quantities describe geometry relative to the consensus. They are not pairwise
distance guarantees, confidence intervals, or evidence for a species boundary.
PCA reconstruction error measures residual outside retained axes, not statistical
prediction confidence. Inspect dispersion before interpreting a local tangent
approximation for widely differing shapes.

## Frozen reference and artifacts

`--reference-run` loads the training consensus, tangent mean and PC loadings.
Each new observation is aligned separately to that fixed consensus and projected
through the same basis. Neither the GPA mean nor PCA is refitted. Landmark IDs,
order, dimensions and configuration must match the saved model. Projection can
process one observation and is independent of the other projection observations.
The report lists overlap with training biological units; training-data
reprojection is allowed but identifiable. Keep evaluation units separate when
using these features in a later supervised model.

A run writes:

- `ingestion/`: unchanged source copies, parsed landmarks and provenance.
- `aligned_landmarks.json`, `.csv` and `aligned_dataset.json`: coordinates in unit-centroid-size units, accepted by the coordinate reader. Source scale factors are not attached to normalized coordinates.
- `shape_arrays.npz`: aligned/tangent coordinates, centroid positions/sizes, rotations, scores and geometric/reconstruction diagnostics.
- `scores.csv`: observation/unit links, source centroid size, angle, radial shrinkage, reconstruction error and PC scores.
- `model.json`, `model.npz`: versioned training configuration/links and hashed numerical parameters. Projection outputs retain the original training links and parameters.
- `gpa_history.json`: fit convergence diagnostics; empty for frozen projection.
- `summary.json`: policies, exclusions, variance/rank, unavailable uncertainty and provenance.

Model loading checks hashes, archive members/sizes, float64 array shapes/finiteness,
centered unit consensus, orthonormal components, tangent constraints and variance
consistency. It uses `allow_pickle=False`. The numerical workspace estimate is
checked against `max_working_bytes`; it is a conservative working-array budget,
not a hard limit on Python parser memory or native SVD workspaces. No pairwise
distance matrix is allocated. A post-creation failure leaves `failure.json` and
partial diagnostics without a top-level completion summary. Inputs and existing
output directories are never overwritten.

```python
from pathlib import Path
from super_deli import load_config
from super_deli.models.shape_config import ShapeConfig
from super_deli.workflows.shape_pca import run_shape_pca

source = Path("examples/shape/train_dataset.json").absolute()
run = run_shape_pca(
    load_config(source), model_config=ShapeConfig(), base_dir=source.parent,
    output_dir=Path("/tmp/super-deli-shape-api"),
)
scores = run.projection.scores
```

Only NumPy is required by the `shape` extra. Configuration/workflow imports remain
dependency-free. Analytic rectangle fixtures specify consensus, raw sizes,
rank-one tangent scores and eigenvalue independently of the implementation.
Tests also compare a two-shape closed form against SciPy's independent
orthogonal-Procrustes solution and cover 2D/3D similarity invariance, reflections,
missingness/repeats, zero variance, safe model round trips and frozen projection.

## Provisional downloaded TPS check

The supplied `res_5184x3456` directory has five 18-point TPS files: 1,084 expert
training records, 248 expert validation records and 720 test records each from
expert, novice and machine digitization. The three test files share image IDs.
Landmark/sliding protocol is unknown, so these were checked **provisionally as
fixed points in source order**, not as validated anatomical or species truth.

The expert training fit converged in four iterations and resolved 32 axes.
Validation and all three test sets were projected without refitting; their image
IDs do not overlap training. All 3,492 digitizations completed. The largest
consensus angle was about 10.734 degrees, with maximum radial shrinkage about
0.584%. These diagnostics support a small local distortion for these particular
checks; they do not validate homology, digitizer accuracy or species hypotheses.
Aggregate counts, hashes and diagnostics are recorded in
[`downloaded_tps_validation.json`](../examples/shape/downloaded_tps_validation.json).
Raw TPS/images remain outside the repository, with source bytes/mtimes preserved.

Reproduce preparation locally, explicitly acknowledging the provisional assumption:

```bash
python examples/shape/prepare_downloaded_tps.py "$HOME/Downloads/res_5184x3456" \
  --assume-fixed-order --output-dir /tmp/super-deli-tps-prepared
super-deli shape-pca /tmp/super-deli-tps-prepared/dataset.json \
  --model-config /tmp/super-deli-tps-prepared/shape.json \
  --modality train_landmark_configuration_expert --output-dir /tmp/super-deli-tps-train
super-deli shape-pca /tmp/super-deli-tps-prepared/dataset.json \
  --model-config /tmp/super-deli-tps-prepared/shape.json \
  --modality val_landmark_configuration_expert --reference-run /tmp/super-deli-tps-train \
  --output-dir /tmp/super-deli-tps-validation
```

Repeat projection for `test_landmark_configuration_expert`,
`test_landmark_configuration_novice` and `test_landmark_configuration_machine`,
using separate output directories. Preparation assigns positional names only;
replace them with the verified anatomical protocol before scientific inference.
