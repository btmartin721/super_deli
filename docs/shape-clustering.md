# Shape PCA Gaussian mixtures and bootstrap stability

The separate [integrative workflow](integrative-clustering.md) combines shape
profiles with other explicitly linked modalities.

```bash
python -m pip install -e '.[dev,clustering]'
super-deli cluster-shape examples/shape/clustering/dataset.json \
  --model-config examples/shape/clustering/model.json \
  --output-dir /tmp/super-deli-shape-clusters
```

Python 3.11/3.12 supports one TPS or landmark-coordinate CSV modality per run.
Choose a new output directory, and use `--modality NAME` when several coordinate
modalities are declared. Image/marker outputs must first supply reviewed,
homologous coordinates under the existing coordinate contract.

## Model and assumptions

The workflow reuses [GPA and shape PCA](shape-pca.md) and fits full covariance
Gaussian mixtures to a **fixed, prespecified number of shape PCs**. Scores are
neither standardized nor whitened. Centroid size, IDs, labels, image metadata,
and other modalities do not enter the mixture features. Size normalization
does not correct allometric shape variation.

Minimum BIC selects among size-admissible candidates K=1 through `max_k`.
The mixture has one covariance matrix per component, regularized by adding
`reg_covar` to its diagonal. This is in squared unit-centroid-size score units.
The API uses [scikit-learn GaussianMixture](https://scikit-learn.org/1.6/modules/generated/sklearn.mixture.GaussianMixture.html).
Full covariances permit correlations among PCs and avoid the diagonal model's
dependence on rotations within a retained PCA subspace. A truncation through
tied eigenvalues can still produce an unstable retained subspace.

```json
{
  "schema_version": 1,
  "assume_independent_units": true,
  "n_pcs": 2,
  "max_k": 4,
  "min_cluster_size": 5,
  "n_bootstrap": 100,
  "n_init": 10,
  "max_iter": 500,
  "tol": 0.001,
  "reg_covar": 0.000001,
  "assignment_threshold": 0.0,
  "random_seed": 0,
  "n_threads": 1,
  "max_pairwise_bytes": 268435456,
  "shape": {"schema_version": 1}
}
```

Only `schema_version` and the explicit independence assumption are required.
Unknown fields, unsupported types and nonfinite values fail. `shape` accepts
the existing shape policies, except that repeated units must be rejected and
`n_components` must be null or equal `n_pcs`. The effective shape model always
retains exactly `n_pcs`; insufficient rank fails instead of changing the feature
dimension. BIC values are compared only within the same fitted PC space, never
across dimensions or different bootstrap fits.

`min_cluster_size` must exceed `n_pcs`. Candidates with insufficient distinct
units are recorded as infeasible; fitted candidates with too few distinct hard
members per component are recorded as inadmissible. This is a pragmatic size
constraint, not a test of covariance identifiability. Regularization and multiple
initializations reduce numerical problems without guaranteeing a global optimum
or the correct number of biological groups. Any feasible candidate's numerical
failure aborts that search, preventing selection from an accidentally reduced
candidate set. Selection can prefer several components along a continuous cline
or within one non-Gaussian group.

Missing coordinates follow `shape.missing_policy`: reject, or explicitly exclude
whole incomplete observations and record them. No landmarks are imputed or
resampled independently. Repeated digitizations of the same biological unit
must be aggregated under a defensible protocol before this checkpoint. The
independence declaration is an assumption the software cannot verify; related
individuals, spatial dependence or sampling clusters may violate it.

## Resampling and uncertainty

Each ordinary nonparametric bootstrap draws N complete biological units with
replacement from the retained N units. Duplicated draws retain all landmarks
and enter the fit with their multiplicities. Component-size checks count
distinct original units. Each replicate refits the GPA consensus, tangent plane,
PC loadings, all candidate mixtures and BIC selection. It then projects **all
original units** through that replicate's frozen shape model and predicts their
mixture membership. Thus the evaluated partition always covers the same units,
even when K or the PC basis changes. No component matching across bases is needed.

Pairwise coassignment is the fraction of requested fits assigning two original
units together. Exact cluster recovery requires the identical original-unit set
to appear as a predicted component. Both ignore arbitrary label permutations.
These are defined pipeline-sensitivity statistics, not implementations of fpc's
Jaccard statistic or its interpretation thresholds. The distinction between
stability and validity is also discussed in the
[fpc clusterboot documentation](https://cran.r-universe.dev/fpc/doc/manual.html#clusterboot).

The main statistic includes predictions for units used in each fit and can be
optimistic. An additional **out-of-bag (OOB)** diagnostic includes a pair only
when neither unit appeared in that draw. Pair-specific requested and successful
denominators are saved. Small counts are noisy; zero-denominator pairs have NaN
arrays and no estimated frequency. OOB stability conditions on a different
training sample for each pair; it is not external validation or a calibrated
confidence interval. No sample is generated or labeled using presumed taxa.

All requested draws are attempted once. Seeds, training indices, OOB indices,
K-search diagnostics and successful predicted partitions are recorded. Rank,
GPA, mixture or original-unit projection failures remain in the requested
denominator. For s observed coassignments and f failed outcomes among B requested
draws, identification bounds are `[s/B, (s+f)/B]`. For OOB pairs, B and f are
pair-specific. Self-coassignment bounds are one for the main statistic.
These bounds describe unknown failed outcomes, **not confidence intervals**.
If any replicate fails, the main coassignment frequency and exact-recovery
frequencies are withheld; successful-subset frequencies are not substituted.
OOB frequency is withheld for each pair affected by a failed draw. Bootstrap K
counts sum to successful draws and are not species-count probabilities.

`n_bootstrap=0` explicitly reports uncomputed stability. Defaults provide 100
draws; the 20-draw example is a bounded software check. Finite bootstrap counts
have Monte Carlo error; choose the count for the intended analysis. Conditional
GMM membership is reported separately and can be overconfident under model
misspecification. `assignment_threshold` only makes low-membership assignments
unresolved; it never changes the fitted partition used for stability.

Mixture components are candidate shape groups. There is no species-boundary
null model, p-value, calibrated species probability, confidence interval,
measurement-error model or multivariate allometry correction in this checkpoint.

## Artifacts and API

| Artifact | Contents |
| --- | --- |
| `shape/` | Reused staged ingestion, aligned coordinates, scores, GPA history, fitted shape model and provenance |
| `mixture.npz` | Scores, zero-based canonical labels, membership, weights, means, covariances, precision Cholesky factors |
| `assignments.csv` | Observation/unit links, candidate cluster, thresholded assignment, reason and conditional memberships |
| `candidates.json` | Point-fit K search and size/convergence diagnostics |
| `bootstrap_replicates.json` | Seeds, unit draws, OOB indices, diagnostics, partitions or failures |
| `stability.npz` | Main counts/frequencies/failure bounds, OOB counts/requested/successful denominators/frequencies/bounds |
| `clusters.json` | Reference component members, exact-recovery counts, frequency and failure bounds |
| `result.json` | Validated assignments/evidence with species p-value explicitly uncomputed |
| `summary.json` | Completion, assumptions, interpretation, versions, input/configuration/model hashes |

Labels and every component-indexed mixture array share canonical order (first
original observation in each hard component); CSV cluster labels are one-based.
Pairwise array axes follow `summary.unit_ids`, also linked to observation IDs.
Load numerical arrays with `allow_pickle=False`. `coassignment` is entirely NaN
after any failed replicate; `oob_coassignment` is NaN for unobserved or
failure-affected pairs. Counts and bounds retain the information needed for
auditing. No stability archive is written when resampling is disabled.

New directories and exclusive writes prevent overwrites. A point-fit failure
writes top-level `failure.json` without a top-level completion summary; partial
shape diagnostics remain available. A completed point result may have unavailable
stability. A conservative `128*N*N` byte budget guards dense pairwise work in
addition to the shape workspace budget. Retained replicate records also use
O(B*N + B*Kmax) storage; choose B and N accordingly.

```python
from super_deli import load_config
from super_deli.models.shape_cluster_config import ShapeClusterConfig
from super_deli.workflows.shape_gmm import run_shape_gmm

summary = run_shape_gmm(
    load_config("examples/shape/clustering/dataset.json"),
    model_config=ShapeClusterConfig(assume_independent_units=True),
    base_dir="examples/shape/clustering",
    output_dir="/tmp/super-deli-shape-api",
)
```

The optional numerical API is `models.shape_gmm.fit_shape_gmm(coordinates,
unit_ids, config=...)`, with N-by-L-by-D complete coordinates. Ingestion,
exclusions, identity handling and export belong to the workflow. The mixture
selector and numerical-fit lock are shared with the genomic baseline; shape
unit bootstrap and genomic locus-block bootstrap retain separate estimands.

The seeded 64-unit example stores latent generating groups separately in
`truth.json`, which is neither a predictor nor staged as input. Tests check
known 2D/3D groups, a single-group simulation, invariance, reproducibility,
refits, failure accounting, OOB denominators and parameter reconstruction.
These are bounded behavior checks, not false-positive calibration or empirical
species truth. The downloaded TPS series from the previous checkpoint remains
provisional with unknown fixed/semilandmark protocol and supplies no species
truth; no species labels are inferred from filenames or digitizer identity.
