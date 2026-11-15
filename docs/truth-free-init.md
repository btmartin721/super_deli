# Opt-in truth-free GMM initialization

```bash
super-deli truth-free-init-config analysis.full-retention.json \
  --arm combined \
  --output-config analysis.diverse-starts.json

super-deli analyze-matched-multimodal dataset.json \
  --analysis-config analysis.diverse-starts.json \
  --output-dir run-diverse-starts
```

The first command is fit-free. It changes exactly one named integrative arm in
an already declared full-retention plan and writes a new file. Original plans
and existing outputs are preserved. Other arms remain unchanged. For an ordinary
plan, first use [full-retention-config](full-retention.md) with an explicit
compatible minimum size.

## Fixed policy

The explicit initialization_policy=diverse_v1 declaration requires
fusion_policy=retain_all_v1 and n_init=100. For every feasible K it fits these
three batches with the existing public scikit-learn GaussianMixture estimator:

| Batch | Initialization | Starts |
|---:|---|---:|
| 0 | kmeans | 34 |
| 1 | k-means++ | 33 |
| 2 | random_from_data | 33 |

The batch with the largest converged lower bound wins, with batch order breaking
ties. This happens before the unchanged distinct-original-member size check.
It does not select the best *eligible* alternative start. Each batch must return
a finite converged fit; numerical failure withholds the search instead of
replacing starts or returning K=1. BIC then selects among eligible K candidates.
No generating labels or reference assignments enter fitting. K is selected
within the configured search range.

The first batch uses the ordinary per-K seed. Other batch seeds are derived
deterministically from that seed, their index and the fixed policy constant
20261115. Candidate JSON records batch methods, start counts, seeds, iterations,
lower bounds, log likelihoods and the chosen batch. The total start count is a
budget, not a claim that all methods have identical runtime.

## Python API

```python
from super_deli.models.full_retention import full_retention_model
from super_deli.models.initialization import truth_free_model
from super_deli.models.integrative import fit_integrative, project_integrative


def fit_and_project(inputs, base_model, new_inputs, new_unit_ids, *, min_cluster_size):
    retained = full_retention_model(base_model, min_cluster_size=min_cluster_size)
    model = truth_free_model(retained)
    fitted = fit_integrative(inputs, config=model)
    projected = project_integrative(new_inputs, new_unit_ids, fitted.pipeline)
    return fitted, projected
```

The truth_free_analysis helper similarly changes one named arm of a
MultimodalAnalysisConfig. Frozen projection and paired whole-specimen bootstrap
use the existing native pipeline. Every draw resamples ingested raw arrays and
refits preprocessing, GPA, view PCA, normalization, joint PCA and the full K
search. SNPio reads the original genomic inputs once per native workflow;
bootstrap draws do not rewrite/re-read genotype files. Multiplicities affect the
fit while minimum component sizes count distinct original individuals.

## Scope and preserved behavior

The policy supports complete, explicitly matched specimen views with shared
covariance, BIC and full retention of the declared per-view PCs. It adds no
missing-view, grouped-sampling, categorical-trait or raw-image landmark model.
Genomic source formats and optional population maps retain their existing SNPio
support. Ecology must remain individually measured and explicitly linked.

The default initialization_policy=legacy is omitted from serialized old plans.
Ordinary joint-PCA/full-covariance defaults and historical p-values remain
unchanged. Explicit legacy 100-start configurations keep their former meaning.
The new policy remains excluded from significance workflows. Memberships and
conditional bootstrap counts are not species probabilities or significance.

## Numerical audit boundary

The validation captures all three batches and independently checks densities,
posterior probabilities, likelihood and BIC with unchanged tolerances. Saved
hard labels must equal the argmax of their saved posterior for every batch.
Independent hard labels must also agree exactly for every batch used at each K,
including candidates that do not win the final K search, and for all legacy fits.

Unused initialization batches can have almost identical components. Their
independent hard-label ambiguity is recorded separately; it does not certify
unique labels and cannot affect component eligibility or reported assignments.
Strict audit behavior remains the default outside this explicitly scoped
validation. The original guard failure and approved audit-only amendment are
retained locally with unchanged scientific arrays.

See [completed validation and decision](truth-free-init-results.md) for recovery,
control splitting, stability, compute costs and limits.
