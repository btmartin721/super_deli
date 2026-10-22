# Shared-covariance integrative clustering

```json
{
  "schema_version": 1,
  "assume_independent_units": true,
  "target_level": "specimen",
  "covariance_type": "tied",
  "n_pcs": 2,
  "max_k": 4,
  "min_cluster_size": 5,
  "n_init": 10,
  "n_bootstrap": 100,
  "random_seed": 71323,
  "views": [
    {"modality": "genomic", "kind": "genomic", "ploidy": 2},
    {"modality": "morphological", "kind": "morphological"},
    {"modality": "ecological", "kind": "ecological"}
  ]
}
```

Save the model configuration separately from the dataset declaration, then run:

```bash
super-deli cluster-integrative /path/to/dataset.json \
  --model-config /path/to/shared-model.json --output-dir /path/to/new/result
```

Python 3.11/3.12 and `super-deli[genomics,clustering]` support this example.
Genomic input continues through SNPio with an optional population map. The
covariance option also works with other supported view subsets, morphometric
views and explicit group aggregation; those input contracts are unchanged.

## Method and assumptions

`covariance_type="tied"` fits separate component means and mixture weights with
one shared full covariance matrix. Correlations among joint PCs are allowed;
all components must share their within-component covariance. This can reduce
estimation variance and the BIC cost of additional groups when that restriction
is appropriate, but can underfit groups with unequal covariances. It does not
change the PCA representation, modality weights, missingness policy, EM starts,
regularization or hard minimum component size.

For K components in d score dimensions, the shared-covariance family has
`K*d + d*(d+1)/2 + K - 1` parameters. The original full-covariance family has
`K*(d + d*(d+1)/2) + K - 1`. At d=2 and N=64, the extra BIC penalty for K=4
versus K=2 is 24.9533 with shared covariance and 49.9066 with separate full
covariances. BIC is computed from the fitted mixture likelihood with the
appropriate family-specific parameter count. It remains an approximate
model-order criterion for regularized finite mixtures, without a general
recovery guarantee.

The default remains `"full"`. Omitting the option or explicitly choosing full
covariance preserves the original version-one serialized configuration and
nested calibration configuration hashes. Shared covariance is opt-in and
appears explicitly in saved configurations, model identifiers and summaries.
There is no automatic choice between covariance families and no selection
based on generating labels. The existing complete K=1..4 search and failure
policy apply within the selected family.

The implementation delegates EM to the supported
[scikit-learn GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html).
Exported `mixture.npz` retains one dense covariance per component for uniform
reconstruction; shared matrices are repeated and must be identical. They still
count as one covariance matrix in BIC. Model summaries identify the family.

## Stability and significance

Paired independent biological-unit resampling refits preprocessing, view PCA,
profile normalization, fusion, joint PCA and the selected covariance-family
GMM. Group members and locus eligibility remain conditioned on the observed
data, as in the existing workflow. Cluster support uses label-invariant exact
unit-set recovery and coassignment frequencies. OOB coassignment has
pair-specific requested/successful denominators. Failed draws are never
replaced: complete frequencies are withheld where required and identification
bounds retain the missing outcomes. These bounds are not confidence intervals.
In-bag stability can be optimistic; OOB sample sizes may be sparse.

Current score, reference-pipeline, empirical continuous, grouped and joint
significance workflows explicitly reject shared-covariance input. Their
published results and full-covariance behavior are preserved. Supporting a new
search family in those tests requires a separate calibration checkpoint.
Membership probabilities, stability and BIC are not species-boundary p-values.

## Frozen paired validation

```bash
super-deli benchmark-shared-covariance examples/shared-covariance/study.json \
  --source-artifacts /path/to/integrative-recovery-20261021 \
  --output-dir /path/to/new/shared-study --workers 4
super-deli analyze-shared-covariance /path/to/new/shared-study \
  --output-dir /path/to/new/shared-audit
```

The configuration pins the previous diagnostic's complete study-file
inventory, summary and source-study declaration. Full source artifacts are
required; portable public tables alone cannot replace them. Both output
directories must be new and separate from the source.

The prescribed six cohorts each contain 25 observations: archived four-group
N=64, fresh four-group N=64, fresh four-group N=256, zero-association N=64,
unequal-variance four-group N=64, and continuous-cline N=64. Fresh master seed
20261030 and cohort-specific streams are independent of the prior diagnostic.
All data use the existing raw genomic–phenotypic generator, except the
prespecified unequal-variance stress: phenotype residual SDs are multiplied by
0.5, 0.8, 1.2 and 1.8 across the four generating groups. This explicitly
violates the shared-covariance assumption.

The two families receive the same raw inputs, two PCs/view and jointly, equal
view weights, K=1..4, minimum size five, ten EM starts, seed 71323, maximum 500
iterations, tolerance 0.001 and regularization 1e-6. Every point reruns SNPio
and the complete integrative pipeline. Their joint scores must match exactly.
The first five fresh four-group observations at each sample size receive 25
unit-bootstrap draws per family with paired random streams. Thus 300 point
workflows and 500 bootstrap refits are planned. No truth initialization,
parameter tuning, replacement attempts or significance reruns are included.

Truth is isolated in evaluation files. ARI and linear axis recovery apply only
to the constructed four-group cohorts. Zero-association and cline cases report
selected-K behavior, without invented discrete truth for the cline or claims
of species-test false positives. Pointwise Wilson intervals describe binomial
simulation uncertainty and are unadjusted across the prespecified cases.
Paired ARI changes and gains/losses in K=4 accompany separate family results.
The stability subset is a software/behavior check with only five observations
per sample size, not a comprehensive stability validation study.

The separate read-only auditor regenerates raw inputs, identities and labels;
checks all inventories, settings, seeds, paired scores and archived baseline
replays; reconstructs fusion/PCA projection and selected dense-mixture
likelihood/BIC/membership; verifies identical shared covariance matrices and
independent ARI/axis calculations; and rebuilds bootstrap sampling,
distinct-unit constraints, K selection, partition counts, OOB denominators,
exact-set support and failure bounds. It refits no model. View PCA fits and
nonselected/bootstrap component likelihoods remain preserved diagnostics,
rather than independently evaluated component fits.

The default full-covariance model is retained regardless of the study outcome.
The completed report records observed improvements and any failures or stress
limitations. Wider biological validation and shared-family significance
calibration remain separate checkpoints.
