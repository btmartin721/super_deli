# Full-covariance grouped power and recovery diagnosis

This diagnostic checkpoint explains weak departure detection and four-group recovery
under the completed full-covariance group-mean PCA calibration. It changes no
production model, default, calibration method or historical p-value. It adds
`diagnose-full-grouped` and a separate read-only `analyze-full-grouped` auditor.

## Frozen design

The source is the checksum-pinned `full-groupmean-null-20261023` study. All 175
archived tests and their 99-draw candidate logs are retained and checked. Only
the 25 complementary four-group observations enter the paired point grid, alongside
25 fresh observations each with 64 and 256 independent groups. The simulator is
unchanged: four traits/view, fixed non-informative unequal morphology counts of
1–4, complete continuous specimen morphology and one ecology row/group.

| Arm | Morphology PCA training | Joint PCs | GMM covariance | EM starts |
| --- | --- | ---: | --- | ---: |
| baseline | Raw group means | 4 | Full | 10 |
| pcs_2 | Raw group means | 2 | Full | 10 |
| pcs_3 | Raw group means | 3 | Full | 10 |
| starts_100 | Raw group means | 4 | Full | 100 |
| shared_4 | Raw group means | 4 | Shared | 10 |
| source_4 | Specimen rows | 4 | Full | 10 |

Every arm keeps two PCs/view, equal view weights, K=1..4, minimum cluster size 5,
model seed 71323 and the other point settings fixed. Master seed 20261112 and all
source pins/grid settings are written before fresh generation. The plan calls for
450 point pipelines, with at most four independent one-thread workers. Counts,
identities and input arrays are paired within each dataset. The source-row arm
changes both weighting and the distribution used to train morphology PCA; it is
not a pure weighting intervention.

Separately, 225 truth-initialized full-covariance EM fits use baseline scores: one
K=4 partition and the two generating K=2 binary partitions for every dataset.
These forced-K fits are optimization diagnostics; truth never enters the
unsupervised pipelines or configuration selection. Their constraints, likelihood,
BIC and label agreement are reported even when they cannot compete by BIC.
There are no new significance or stability draws. An earlier numerical issue and
its retained failures remain separate and unmodified.

## Quantities and audit

The analysis traces generating axes through raw group means, source-row scores,
aggregated profiles, fusion and joint PCA, retaining loadings, counts and covariance
eigenvalues/condition numbers. It reports selected K, ARI, eligibility/convergence,
within-score-space likelihood gains and BIC parameter costs. For full covariance,
`q=(K-1)+K*d+K*d*(d+1)/2`; shared covariance uses
`q=(K-1)+K*d+d*(d+1)/2`. Absolute BIC values are never compared across different
score dimensions. The shared/full contrast uses the same four-dimensional scores;
no family is selected for deployment from these results.

The archived-tail analysis independently reconstructs every retained T from
candidate logs and the exact upper-tail count, including ties. `T=0` implies p=1
because K=1 belongs to the search. All incomplete fits retain their error/attempt
records; no fit or observation is replaced. Failed baseline fits leave their
truth diagnostics unattempted. The primary archived tests remain approximate
plug-in model-departure tests with nonregular boundary caveats. Their p-values
do not identify K, memberships or species boundaries.

The no-refit auditor checks immutable source/artifact hashes, raw regeneration,
identity alignment, seeds, settings, group-mean PCA moments/eigen-equations and
projections, fusion, selected Gaussian-mixture density/membership/BIC, independent
contingency ARI and pseudoinverse axis recovery, and truth-based initialization
and density. Nonselected likelihoods are inferred from retained BIC logs, not
independently refitted or recomputed from unretained parameters.

K=4 rates receive pointwise Wilson 95% intervals; mean ARI has an independent-dataset
Monte Carlo standard error. Paired changes are descriptive. These summaries are
unadjusted across the diagnostic grid and conditional on availability; retained
failure bounds are not confidence intervals. Alternative arms get no new p-values
or calibrated power estimates. Results under one constructed four-group law do
not establish general robustness or empirical species validity. No winning
configuration becomes a default.

## Reproduction

Use Python 3.11 or 3.12 with the clustering extra and pinned source artifacts:

```bash
super-deli diagnose-full-grouped examples/full-grouped-diagnostics/study.json \
  --source-artifacts /path/to/full-groupmean-null-20261023 \
  --output-dir /path/to/new-diagnostics --workers 4
super-deli analyze-full-grouped /path/to/new-diagnostics \
  --output-dir /path/to/new-audit
```

All outputs require new directories. The source study and existing checkouts stay
unchanged. The portable [results](full-grouped-diagnostics-results.md) include
per-point, candidate, optimization and archived-tail tables; complete raw inputs,
PCA/GMM arrays, logs and the frozen runtime remain external. Local preservation is
not public raw deposition or a self-contained runtime image.
