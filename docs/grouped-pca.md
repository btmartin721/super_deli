# Group-mean morphology PCA for integrative clustering

`pca_fit="target_means"` is an opt-in view policy for complete continuous specimen
morphology with explicit population/locality links and `aggregation="mean"`.
It fits preprocessing and PCA on one raw morphology mean per retained target.
This gives each group equal influence on feature standardization and PCA directions,
rather than weighting groups by their specimen counts. Ecology remains one directly
measured row per group. Genomic, count, missing-cell and landmark PCA are outside
this new policy. Their existing source-row workflows are unchanged.

The default is `pca_fit="source_rows"`. Default joint dimension remains two; default
covariance remains full. Default policy keys are omitted during serialization so
legacy configuration JSON and hashes remain compatible. The option is accepted
only for upward specimen-to-population/locality morphology aggregation; categorical
features, imputation and count transforms are not added.

## Geometry, projection and uncertainty

Raw means are calculated from every declared member of each shared target. Fit
feature means/scales and fixed two-PC morphology on this group matrix, then project
source rows with those parameters and average their scores within targets. For
complete continuous data this equals projecting each raw mean. Averaging scores is
linear; the changed training distribution determines new scales/directions.

Paired whole-target bootstraps retain every observed member and its group count.
A repeated target contributes its raw mean once **per draw occurrence** when PCA
is refitted; it does not contribute one row per specimen or lose multiplicity by
collapsing duplicate draws. View preprocessing/PCA, group profiles, fusion scaling,
joint PCA and the complete GMM K search are refitted. Projection uses the frozen
training parameters. Diagnostics record training target IDs/counts, source member
indices and target-mean PCA parameters for independent reconstruction.

Clustering weights each target profile equally. This does not estimate a latent
error-free group phenotype: groups with fewer specimens have noisier means, and
ecology measurement error is not modeled. Bootstrap stability conditions on fixed
members/counts and measures algorithmic reproducibility under target resampling.
It is neither species probability nor a significance test. Failed replicates retain
requested denominators and identification bounds; OOB pairs have separate counts.

All significance adapters reject the new policy until it receives separate
calibration: grouped, matched continuous, raw reference, joint genomic–phenotypic,
and frozen-score workflows. Historical p-values and archived evidence remain
unchanged. Full covariance and two joint PCs remain available and remain defaults.

## Example

The morphology view can declare:

```json
{
  "modality": "morphological",
  "kind": "morphological",
  "aggregation": "mean",
  "pca_fit": "target_means",
  "n_pcs": 2
}
```

Use `examples/grouped-pca/model.json` with a dataset containing the matching
morphological/ecological modality names and explicit group links. That example is
an opt-in four-joint-PC, shared-covariance configuration with stability enabled;
it is not an automatically selected or universal recommendation.
`examples/grouped-pca/source_rows_model.json` retains source-row PCA with four
joint PCs for the comparison supported by the recovery evidence.

## Frozen validation

The prespecified four arms are source-row/group-mean morphology PCA crossed with
two/four joint PCs. Every view retains two PCs, equal weights, K=1..4, minimum size
5, 10 EM starts, 500 iterations, tolerance 0.001, regularization 1e-6, one thread
and model seed 71323. No generating labels enter fitting or settings selection.

Study declaration: 25 archived four-group N=64 observations from the frozen grouped
diagnostic and 25 fresh datasets each for four-group N=64, four-group N=256,
independent Gaussian, correlated Gaussian, unequal-variance four groups and a
continuous cline. Counts remain independent 1–4 specimens/group; each ecology view
has one group row. There are **175 datasets and 700 point workflows**. The first
five fresh four-group datasets at each size attach 25 stability draws/arm to their
existing point fits: **1,000 planned bootstrap refits**, without extra point fits.
Fresh master seed **20261106** is fixed before generation.

The original complementary/independent/correlated generator is unchanged.
Unequal-variance stress scales residuals around the four generating centers by
fixed SD multipliers 0.5, 0.8, 1.2 and 1.8, violating shared component covariance.
The cline starts from correlated Gaussian observations and adds an independent
uniform(-1,1) group coordinate times 3*[1,0.7,0.5,0.3] to both views. It has no
discrete generating partition. These stress laws are versioned in the new study
module rather than changing historical simulators.

```bash
super-deli benchmark-grouped-pca examples/grouped-pca/study.json \
  --source-artifacts /path/to/grouped-recovery-artifacts \
  --output-dir /path/to/new-study --workers 4
super-deli analyze-grouped-pca /path/to/new-study \
  --output-dir /path/to/new-audit
```

Pin source declaration/summary/inventory/runtime hashes and verify the entire
archived study. A declared source-change allowlist covers this implementation;
all other archived scientific modules must match. Reproduce archived baseline
scores, candidate status/BIC and assignments before interpreting comparisons.
Freeze the wheel, settings and seeds before fresh generation. Retain every failure
without replacement. Previously archived numerical failures remain untouched;
repairing their matrix-runtime issue is outside this checkpoint.

The no-refit audit regenerates raw observations/identities/truth, checks complete
inventories/ledgers, and verifies PCA means/scales/eigen-equations, projection,
profiles, fusion and selected GMM likelihood/BIC/membership independently. It checks
target-mean bootstrap PCA parameters with recorded multiplicities, source member
indices, seed streams, distinct-unit constraints, exact-set/coassignment counts,
OOB denominators and failure bounds. Source-row bootstrap PCA and nonselected GMM
parameters are not independently refitted. Numerical convergence does not certify
a global optimum.

Evaluate K and ARI separately, with truth used after fitting. Candidate likelihood
and BIC decompositions are compared only within their fixed score spaces, never as
absolute BIC across arms. Wilson intervals across independent datasets are
pointwise and unadjusted for the four-arm grid. Cluster selection under Gaussian
or clinal laws without discrete groups is descriptive; it is not species-boundary
significance or a calibrated type-I error rate. No new default is promoted.
