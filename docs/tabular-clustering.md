# Numeric morphology and ecology PCA/GMM baseline

To combine this data with genomic or shape evidence, use the separate
[integrative workflow](integrative-clustering.md).

```bash
python -m pip install -e '.[dev,clustering]'
super-deli cluster-tabular examples/tabular/clustering/dataset.json \
  --model-config examples/tabular/clustering/morphology_model.json \
  --modality traits --output-dir /tmp/super-deli-morphology-clusters
super-deli cluster-tabular examples/tabular/clustering/dataset.json \
  --model-config examples/tabular/clustering/ecology_model.json \
  --modality environment --output-dir /tmp/super-deli-ecology-clusters
```

Python 3.11/3.12 supports one morphology or ecology table per run, using the
existing [typed CSV/TSV contract](tabular-ingestion.md). Select `--modality NAME`
when several tables exist. Outputs use a new directory and preserve source
files, row identity, biological level, feature definitions, and provenance.
The `clustering` extra has no SNPio or image-codec dependency.

## Features and biological units

Continuous measurements and nonnegative integer counts are supported. Lengths,
masses, trait measurements, temperature, precipitation or other environmental
measurements can be declared as analytical features. These require defensible
sampling, comparable measurements and a consistent protocol. Units are retained
as metadata; no physical unit conversion or covariate adjustment is inferred.

`feature_names=[]` selects every column with `role="feature"`, in declaration
order. An explicit list selects those analytical columns in exactly that order.
Unknown names or covariates in the list fail. Selecting any binary, ordinal or
categorical feature also fails; this baseline does not assign numerical distances
to category labels. Use an explicit numeric subset for mixed tables. Other
features and covariates remain in the ingestion artifact and are listed as
nonselected columns. IDs, taxonomic truth and covariates never enter PCA.

There must be one observation per independent biological unit. Repeated
digitizations, repeated trait measurements or repeated environmental readings
require aggregation under an external, explicit protocol. `specimen`,
`population` and `locality` levels are preserved in assignments and evidence.
Locality clusters describe environmental groups at those localities; they do
not assign species to individual specimens. No cross-table matching or fusion
is performed in this checkpoint, even when tables share unit or locality links.

## Configuration and preprocessing

```json
{
  "schema_version": 1,
  "assume_independent_units": true,
  "feature_names": [],
  "missing_policy": "error",
  "scaling": "standard",
  "count_transform": "identity",
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
  "rank_tol": 1e-12,
  "max_working_bytes": 268435456,
  "max_pairwise_bytes": 268435456
}
```

Only `schema_version` and an explicit `assume_independent_units=true` are
required. Unknown fields, invalid types and nonfinite settings fail.

The fitted pipeline, repeated for every bootstrap draw, is:

1. Select the declared continuous/count analytical features. Counts must be
   nonnegative integers exactly representable as float64. Optional
   `count_transform="log1p"` applies `log(1+x)` to count columns only, including
   zero counts. Continuous columns retain their original measurement scale.
2. Fit means on observed transformed values. Exclude all-missing and exactly
   zero-variance columns, recording their names/reasons. Feature filtering is
   learned anew in each draw. At least `n_pcs` usable features must remain.
3. Reject missing cells in retained columns under `missing_policy="error"`.
   Explicit `mean` fills these cells with that fit's transformed-column means.
   A row missing every retained feature fails instead of becoming an entirely
   imputed observation. Ingestion retains original missing cells and masks.
4. Always center. Default `scaling="standard"` divides by the standard deviation
   of the mean-filled training values using denominator N (`ddof=0`). Explicit
   `center` leaves the transformed units unscaled, so large-scale variables can
   dominate PCA. Saved means and scales reproduce this transformation. The
   convention follows [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).
5. Fit full-SVD covariance PCA without whitening, retaining exactly the
   prespecified `n_pcs`. PCA variance uses N-1. The API uses
   [scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
   A requested final singular value at or below
   `rank_tol * max(largest_singular_value, 1)` fails; dimension is never silently
   reduced. No comparison of BIC across PC dimensions is made.
6. Fit full covariance Gaussian mixtures for K=1 through `max_k`, selecting
   minimum BIC among size-admissible candidates. Covariances are regularized by
   `reg_covar` in squared PC-score units. `min_cluster_size` must exceed `n_pcs`.
   Feasibility and hard component sizes count distinct biological units,
   including in bootstrap draws with duplicates. Any feasible candidate's
   numerical/convergence failure aborts that fit's entire K search.

Mean imputation is an explicit approximation and can reduce variance or conceal
missingness mechanisms. Refitting its means does not model uncertainty in each
missing value or fix nonignorable missingness. An all-missing selected column
is excluded even under the `error` policy, with a recorded exclusion. Rank
deficiency, nonrepresentable arithmetic and insufficient sample sizes fail
explicitly. Scaling is sensitive to outliers; there is no automatic outlier
removal, allometry correction or transformation of continuous measurements.
Gaussian components approximate score distributions; count data, non-Gaussian
traits, clines or sampling imbalance can lead to several components within one
biological group. BIC does not test species boundaries.

## Bootstrap evidence

The engine is shared with [shape clustering](shape-clustering.md), while its
pipeline callback refits tabular preprocessing/PCA/GMM. Each draw samples N
whole rows with replacement and retains their multiplicities in the likelihood.
It then projects all original units through that draw's fitted column filter,
imputation means, scaling and PCA before predicting their mixture labels.
All columns of each biological observation stay together; columns are not
treated as independent bootstrap units. Covariance regularization and multiple
initializations do not guarantee a global optimum or biological correctness.

Pairwise coassignment and exact original-unit-set cluster recovery avoid label
matching across replicates. The main statistic includes in-bag predictions and
can be optimistic. The separate OOB statistic uses a pair only when both units
were absent from the training draw and supplies pair-specific requested and
successful denominators. OOB comparisons with sparse counts are noisy; pairs
never observed together out of bag have no frequency estimate.

Every requested draw is attempted once. Failures are recorded with seeds,
training/OOB indices and errors, without retries or successful-subset
substitution. For s observed coassignments and f unknown failed outcomes among
B draws, failure bounds are `[s/B, (s+f)/B]`. They are identification bounds,
not confidence intervals. Any failed draw withholds the main frequency and
exact-set frequencies; OOB frequencies are withheld for failure-affected pairs.
Main self-coassignment bounds remain one. K counts sum to successful draws and
are not species-count probabilities. `n_bootstrap=0` declares uncomputed
stability. Finite bootstrap counts also carry Monte Carlo error; the 20-draw
examples are software checks rather than biological uncertainty calibration.

The assumption of independent biological units cannot be verified here.
Spatial or temporal autocorrelation, related individuals, shared sites, or
sampling groups may require a different sampling model. This checkpoint does
not implement spatial/block/stratified row resampling. In particular, ecological
clustering and its stability are descriptions of sampled environments, not
independent evidence of species boundaries. Conditional GMM membership,
bootstrap frequency and significance remain separate result evidence types.
No species null test, calibrated species probability or confidence interval
is computed. Assignment thresholds affect reported assignments only; stability
uses the full fitted partition.

## Artifacts and API

| Artifact | Contents |
| --- | --- |
| `ingestion/` | Staged source, typed table, source masks, coverage and hashes |
| `preprocessing.json` | Selected order/types, retained/excluded/nonselected columns, policies, imputation count and array hash |
| `preprocessing.npz` | Retained indices, means, scales, PCA mean/loadings/variance/singular values and missing fractions |
| `scores.csv` | Observation and biological unit IDs linked to PC scores |
| `mixture.npz` | Scores, canonical labels, memberships, weights, means, covariances and precision Cholesky factors |
| `assignments.csv`, `result.json` | Candidate and thresholded assignments, conditional memberships, distinct evidence and level |
| `candidates.json` | Point-fit K search diagnostics |
| `bootstrap_replicates.json` | Draw/fit seeds, training/OOB indices, fitted features, imputation counts, candidates, partitions or errors |
| `stability.npz`, `clusters.json` | Coassignment/OOB counts and denominators, exact recovery and failed-outcome bounds |
| `summary.json` | Completion, biological level, assumptions, versions, configuration and input/model hashes |

The numeric arrays use the same non-pickle convention as shape clustering.
Component-indexed arrays share canonical order; labels in NPZ are zero-based
and CSV labels one-based. Pairwise axes follow `summary.unit_ids`. Main
`coassignment` is all NaN after any failure; OOB NaNs denote unobserved or
failure-affected pairs. Counts and bounds remain available. No stability NPZ
is written when disabled. A failed point analysis writes `failure.json` and
no completion summary. Existing output directories are never overwritten.

```python
from super_deli import load_config
from super_deli.models.tabular_cluster_config import TabularClusterConfig
from super_deli.workflows.tabular_gmm import run_tabular_gmm

summary = run_tabular_gmm(
    load_config("examples/tabular/clustering/dataset.json"),
    model_config=TabularClusterConfig(
        assume_independent_units=True,
        feature_names=("length_mm", "head_width_mm", "mass_g", "scale_count"),
        count_transform="log1p",
    ),
    modality_name="traits",
    base_dir="examples/tabular/clustering",
    output_dir="/tmp/super-deli-tabular-api",
)
```

The optional numeric API is `models.tabular_gmm.fit_tabular_gmm(table,
config=...)` for an ingested `TabularTable`. Its returned PCA model supports
`project_tabular_pca(values, model=...)` with numeric columns in the saved
selected-feature order and NaN for missing cells. This checkpoint does not
provide a saved-run prediction CLI. Archives permit independent numerical
reconstruction without loading serialized Python objects.

Separate configurable budgets guard PCA workspaces and dense `128*N*N` byte
pairwise work. Replicate records additionally require O(B*N + B*Kmax) storage.
The process-wide numerical-fit lock is shared with shape and genomic baselines;
callers running unrelated numerical threads should isolate jobs in processes.

The seeded morphology/ecology fixtures contain 64 independent units per table,
continuous and count columns, a nonselected categorical trait and a covariate.
Latent generating labels are in a separate `truth.json`, never staged or used
as predictors. Tests compare scaling/PCA to an independent reference and check
known groups, a single-group simulation, continuous structure, missingness,
bootstrap training-only refits, OOB denominators, parameter reconstruction,
identity levels and source preservation. These checks do not establish empirical
species truth or control false-positive species delimitations.
