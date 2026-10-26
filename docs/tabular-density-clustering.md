# Continuous tabular PCA + HDBSCAN

This opt-in second clustering family proposes density groups from one complete
continuous morphology or ecology table. It preserves all PCA/GMM models,
defaults, legacy configurations and historical p-values. Neither this workflow
nor its benchmark opens any empirical sampling/reference-evidence gate.

## Scope and model

Declare one observation per independent specimen, population or locality. The
analyst must explicitly assert independence; the software cannot verify it.
Select analytical continuous columns; covariates remain excluded. Counts,
categories, missing analytical cells, repeated units and new genomic, shape or
multimodal density adapters are outside this checkpoint. Constant features are
recorded and excluded; the fixed requested PCA dimension must remain identifiable.

The pipeline reuses numeric-table validation, standardization (population SD),
full-SVD PCA and rank checks. Its separate `HDBSCANConfig` does not inherit GMM
selection or calibration parameters. Prespecified defaults are two PCs,
Euclidean distance, `min_cluster_size=5`, `min_samples=5`, `alpha=1`, EOM,
`cluster_selection_epsilon=0`, `allow_single_cluster=true` and one thread.
The scikit-learn convention includes the observation itself in `min_samples`.
See the [supported 1.6 API](https://scikit-learn.org/1.6/modules/generated/sklearn.cluster.HDBSCAN.html).

Nonnegative labels identify candidate density groups; **-1 means unassigned/noise**.
Zero clusters/all noise is a valid successful result. Density membership strength
is the algorithm's within-cluster strength, not a calibrated species probability.
A group count alone is insufficient: inspect assigned coverage and the observations
left unassigned. PCA can lose biologically relevant low-variance signals, and
scaling and density settings materially affect the partition.

## CLI and artifacts

Install `super-deli[clustering]`, declare a standard dataset JSON, then run:

```bash
super-deli cluster-tabular-density dataset.json \
  --model-config examples/density-clustering/model.json \
  --modality traits --output-dir density-run
```

The dataset path resolves its inputs relative to its own directory. Outputs must
be new directories. `summary.json` declares the family, units, level, group/noise
counts, unavailable significance and conditional stability status.
`assignments.csv` retains exact observation/unit IDs, labels, strength and scores.
`point.npz` stores preprocessing parameters, PCA and point labels/strength;
`preprocessing.json` records feature order, exclusions and nonselected columns.
`subsamples.json` and `stability.npz` retain raw draw identities, failures,
per-draw fitted transformations/outputs and denominator arrays. Binary arrays are
numeric and readable with `allow_pickle=False`. Provenance binds configuration,
ingested source hashes, software versions and artifact hashes. Existing artifacts
are never overwritten. A CLI run with incomplete subsampling returns nonzero but
preserves the successful point, successful draws and failure records.

## Conditional subsampling stability

The example requests 25 independent 80% subsamples **without replacement**; each
contains `floor(0.8*N)` whole units. Refit standardization, feature exclusions,
PCA and HDBSCAN from the selected raw rows. Original identities stay unique;
ordinary-bootstrap duplication would change local density and is not used.
There is no invented prediction for absent/new observations.

For each unit, noise frequency divides noise outcomes by successful draws that
included it. For each pair, coassignment divides positive-cluster matches by
successful draws containing both units. Two noise observations never count as a
shared cluster. A diagonal can be less than one because its unit can be noise.
Never-included or never-successful denominators remain unavailable (NaN in
numeric arrays), rather than zero. Inclusion counts, successful counts, coverage,
K/noise frequencies and conditional Bernoulli Monte Carlo SEs are explicit.
These SEs quantify precision over repeated draws conditional on the input cohort
and frozen procedure; they are not biological sampling confidence intervals.

Failed draws are not replaced or assigned K=1. A complete-case conditional
summary is marked incomplete/unavailable when necessary. Worst-case lower/upper
bounds use all attempted inclusions, assigning each failed outcome zero or one.
Stability does not establish species probability or significance. This estimator
has no calibrated p-value; GMM significance adapters reject its family.

## Frozen paired study

`examples/density-clustering/study.json` specifies 300 fresh datasets: 25 per cell
across Gaussian, balanced four-group, unequal-density/variance four-group,
two noisy moons, four groups with background and continuous-cline laws, each at
N=64/256. Two continuous input traits and two PCs preserve the standardized input
geometry, isolating this clustering comparison from dimensional truncation.
Parameters and seed streams are fixed before generation in the local frozen plan.
The approved plan is preserved locally with exact means/covariances/curves.

The comparator reuses the unchanged tabular full-GMM numerical worker, full
covariance, BIC, K=1..4, minimum size five and ten EM starts. Seed 71323 is passed
directly to its existing worker, which deterministically derives candidate seeds.
This is the same numerical baseline, without the public wrapper's additional
seed derivation. HDBSCAN has no fixed K ceiling, so search spaces and model
assumptions differ. No absolute BIC is compared between families.

The first five datasets per cell receive 25 paired identical subsamples per
family. Each family refits its complete numerical pipeline. This gives 600 point
plus 3,000 subsample pipelines, after a separate 624-pipeline full-settings
preflight. Counts describe pipeline runs, not equal computational costs or numbers
of fitted GMM components. All failures and unattempted runs remain accounted for.

Truth is kept outside fitting inputs and used only after fitting. Report group
count, ARI, assigned/noise coverage and split/merge diagnostics. For agreement
among the non-background generating groups, each predicted noise observation is
treated as its own unassigned singleton. Assigned-only ARI is always accompanied
by coverage. Background detection has separate precision/recall; generating
supports overlap. The cline has no discrete truth partition or correct-K/ARI
score. Extra groups under Gaussian controls indicate algorithmic overpartitioning,
not a calibrated significance or species false-positive rate.

Twenty primary correct-K rates receive simultaneous exact 95% intervals using
Bonferroni adjustment. Failed outcomes induce explicit bounds. Other metrics,
paired differences and stability are descriptive; datasets, rather than their
repeated subsamples, define independent benchmark observations. No configuration
or family is promoted automatically from these results.

Independent read-only audits check source hashes, identities, seeds, PCA means/
scales/projections/eigenvalues, all pair distances, selected-GMM density/membership/
BIC, candidate eligibility, draw masks, support denominators and exact rate exports.
HDBSCAN label/strength invariants and known-case tests are checked; the underlying
HDBSCAN implementation is not independently reimplemented by this audit.

Detailed evidence stays local. The measured gate limits newly retained evidence
to 1 GiB, requires 8 GiB free and permits up to four single-thread study workers.
Public payload is limited to 16 MiB of code/docs/invented examples/small summaries.
No empirical fitting, author contact, deep learning, new calibration or default
change is included. See the [results](tabular-density-results.md).
