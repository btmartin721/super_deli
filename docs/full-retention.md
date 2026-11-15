# Opt-in full-retention fusion

The retain_all_v1 policy keeps every selected per-view PC in a matched
specimen-level integrative analysis. The existing full-SVD joint PCA still centers
and rotates the normalized fused coordinates. Its output dimension equals its
input dimension, so it discards none of that selected score space.

This preserves the selected view PCs, rather than guaranteeing preservation of
information already removed by genomic filtering, GPA or per-view PCA. It supplies
candidate clusters and conditional memberships, not species probabilities.

## CLI

Start with a declared shared-covariance (tied), BIC integrative arm in a complete
matched multimodal analysis. Ecological values must be individually measured and
explicitly declared as such in the unified configuration.

```bash
super-deli full-retention-config analysis.json \
  --arm combined \
  --min-cluster-size 9 \
  --output-config analysis.full-retention.json

super-deli analyze-matched-multimodal dataset.json \
  --analysis-config analysis.full-retention.json \
  --output-dir run-full-retention
```

The first command is fit-free. It replaces exactly the named integrative arm and
writes a new configuration; it preserves the source and rejects existing output
files. Singleton reference arms and other combined arms remain as declared.

| Selected views with two PCs each | Retained joint PCs | Smallest accepted minimum cluster size |
|---|---:|---:|
| Two | 4 | 5 |
| Three | 6 | 7 |
| Four | 8 | 9 |

The minimum is explicit because the current native guard requires it to exceed
the joint dimension. Preset creation computes the dimension from selected views
and validates the supplied minimum. Rank-deficient fits remain withheld.

## Python API

```python
from super_deli.models.full_retention import full_retention_model
from super_deli.models.integrative import fit_integrative, project_integrative

def fit_and_project(inputs, base_model, new_inputs, new_unit_ids, *, min_cluster_size):
    model = full_retention_model(base_model, min_cluster_size=min_cluster_size)
    fit = fit_integrative(inputs, config=model)
    projected = project_integrative(new_inputs, new_unit_ids, fit.pipeline)
    return fit, projected
```

Frozen projection uses the fitted view models, normalization and full joint
rotation. Inputs must retain the declared feature, original-locus/allele-coding
and landmark order. Projecting data through the fitted pipeline does not refit
those training transformations. The native parameter exports remain available.

## Supported scope and uncertainty

The tagged policy initially supports two to four complete, unaggregated specimen
views with prespecified shared covariance and BIC. Sources can include SNPio-
ingested diploid genomic data, continuous tables and reviewed fixed-landmark TPS
or coordinate files. Original genomic formats and optional popmaps are unchanged.
All selected views must be explicitly linked to the same individuals.

With n_bootstrap greater than zero, the native workflow resamples whole specimens
with replacement. Every draw refits preprocessing, GPA, view PCA, normalization,
full-coordinate rotation and the complete GMM search. SNPio ingests the observed
source once per native workflow; numerical refits use resampled raw arrays.
Source genomic loci stay fixed. Duplicate draw copies do not increase the distinct
original-member counts used for component eligibility.

Native selected-K counts, label-invariant coassignment, exact point-cluster
recovery and separate OOB pair coverage/failure bounds are preserved. These
measure conditional sampling sensitivity. Draws are repeated measurements from
one dataset, and their frequencies are not species probabilities or significance.

## Compatibility and inference boundaries

The default remains ordinary two-PC joint PCA and full covariance. Old serialized
configurations omit the default fusion-policy field and retain their previous
meaning. Explicit legacy higher-PC configurations remain supported. On the new
policy, the declaration includes fusion_policy=retain_all_v1 and an exact joint
count equal to the sum of view counts. Numerical fitters and start/admissibility
rules are unchanged.

Significance workflows reject the new tagged policy until a separately validated
calibration checkpoint. Historical p-values remain unchanged. Grouped sampling,
missing-view handling, predictive screening and automatic dimension/family/weight
selection are outside this policy's current support boundary.

The [completed validation and N=64 audit](full-retention-results.md) reports
fresh recovery, control splitting, conditional stability and optimization limits.
