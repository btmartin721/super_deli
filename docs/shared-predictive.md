# Shared-covariance predictive-screened grouped AIC

This optional extension supports complete continuous specimen morphology linked
explicitly to populations/localities, plus one continuous ecology row per target.
Independent targets, fixed noninformative specimen counts and group-mean morphology
PCA remain explicit requirements. It reuses the current grouped pipeline and
[predictive screen](predictive-aic.md). Full covariance and BIC remain defaults.
Existing configurations, historical p-values and full-family results retain their
meaning. The [preceding diagnosis](predictive-recovery-results.md) motivates
reducing covariance estimation at small sample sizes; improvement is not assumed.

## Model and fixed selection rule

Use `selection_criterion="aic"` and explicitly choose `covariance_type="tied"`
in an integrative model configuration. Each component has its own mean and weight,
while all components share one full positive-definite covariance matrix. The
full family estimates a separate covariance per component. With K components and
d scores, parameter counts are K*d+K-1+d*(d+1)/2 for tied covariance and
K*d+K-1+K*d*(d+1)/2 for full covariance: 29 versus 59 at K=4,d=4.
Shared covariance can underfit unequal within-group variation.

The extension also permits ordinary grouped AIC with the same explicit tied model,
which supplies the ordinary comparator for the screen. Neither workflow chooses
between covariance families. Choose the family before seeing outcomes; no family
selection, new significance procedure or automatic default promotion is provided.

The screening rule is unchanged: five shuffled whole-target folds, full raw-data
training-only refits, K=1..4 search, full/fold eligibility, strictly positive pooled
validation log-density gain against each fold's K=1, then minimum full-data AIC
with smaller-K ties. The validation pool weights units equally. Source group means,
standardization, view PCA, aggregation, normalization, fusion and joint PCA are all
refit in each training fold. No ecological observation is copied to individuals.

The existing ten-start, best-likelihood-start-before-size-screen behavior remains.
A numerical failure withholds the screened result; it is not a K=1 observation.
Ordinary AIC remains available if only fold screening fails. Membership probabilities
are conditional on a selected fitted mixture, not uncertainty in the screen.
Stability and significance for the new selection rule remain unavailable. Existing
likelihood calibration rejects tied AIC; its earlier full-family p-values are
unchanged. Direct-score shared screening is outside this raw-data checkpoint.

## API, command and exports

Use Python 3.11/3.12 with the `clustering` extra. The existing command accepts an
explicit tied model:

```bash
super-deli cluster-integrative-predictive /absolute/dataset.json \
  --model-config /absolute/tied-model.json \
  --screen-config /absolute/screen.json \
  --output-dir /absolute/new-output
```

All inputs and outputs remain identity-linked and non-overwriting. Candidate
arrays retain the existing (K,d,d) expanded covariance layout: tied matrices are
identical across components. Model/run metadata explicitly declare the covariance
family. `selected_model.npz` additionally stores scalar `covariance_type="tied"`
for this extension, with ordered means/weights/covariances and component order.
Full-family candidate/selected-array schemas remain unchanged. Independent audits
verify identical shared matrices, family-specific criteria, memberships and reloads.

## Frozen paired validation

The approved design has 350 independent observations, 25 in each of 14 cells:

| Role | Laws | Localities | Datasets |
| --- | --- | --- | ---: |
| Controls | Independent, correlated, zero-conditional hierarchical Gaussian | 64 and 256 | 150 |
| Alternatives | Complementary four groups, signals 3 and 4.5 | 64 and 256 | 100 |
| Stress | Unequal-variance four groups | 64 and 256 | 50 |
| Stress | Heavy tails and continuous cline | 64 | 50 |

All raw laws are unchanged. Retain 1–4 morphology specimens/locality, one ecology
row/locality, four traits/view, two PCs/view, four joint PCs, equal weights,
minimum distinct hard size five, ten starts, max_iter=500,tol=0.001,reg_covar=1e-6
and model seed 71323. Both families share data, identities/counts, partition and
model seeds, with independent complete pipeline fits and independently verified
identical transformations/scores. Ordinary/screened choices share candidates.
Labels are used only after fitting. Unequal-variance/cline/heavy-tail stress is
separate from species or type-I error interpretation. Signal 1.5 has no fresh
shared-family validation in this checkpoint.

Each family needs one full plus five fold searches, for 4,200 searches across
350 observations. Add 25 verified archived full-family six-search replays, five
each from N=64 signals 1.5/3/4.5, N=256 signal 3 and N=64 independent Gaussian:
**4,350 main searches**. A separate full-settings storage preflight uses one pair
per cell: **168 searches**. There are no oracle fits, 100-start arms, calibration
or stability draws. Development/tests/installed-smoke fits are separate.

Independent external sets have 2,048 localities/observation. These outcomes never
set folds, thresholds, models or family choices. Compare density gains within the
same learned map; do not compare absolute density/AIC across separately learned
maps. Study and probe seeds, settings and source fingerprints are frozen before
fresh generation and checked against earlier seed domains.

The primary family contains 40 rates: ordinary/screened K>1 for two families in
six control cells, and K=4 recovery in four alternative cells. Give exact pointwise
and Bonferroni simultaneous 95% intervals, dataset-level MCSE, denominators and
withheld-outcome bounds. Thirty paired comparisons cover screening within each
family and tied-screen versus full-screen in each primary cell, with simultaneous
conservative exact discordance intervals. Stress K distributions, ARI, predictive
gains, gate failures, spectra and BIC remain descriptive. Folds are not independent
datasets; no fold-independence standard error or CV p-value is supplied.

## Reproduction, audit and local storage

Frozen configurations and the installed smoke are in `examples/shared-predictive/`.
Commands are `benchmark-shared-predictive`, `analyze-shared-predictive`,
`preflight-shared-predictive` and `replay-shared-predictive`; use `--help` for paths.
The installed smoke runs eight smaller raw-data pairs (96 searches), then numerical
and separate standard-library audits.

Independent numerical checks reconstruct raw laws, IDs/counts, folds, preprocessing,
PCA, projections, covariance geometry, likelihoods/AIC/BIC, membership, eligibility,
gates, selections and selected-model round trips. The standard-library checker
verifies archive/member bytes, saved criteria/gains, paired maps and accounting.
Audits refit no models. Stable Cholesky external-density checks retain the earlier
high-precision regression and original tolerances. Failures and unattempted fits
remain retained without replacement or denominator removal.

Full studies require a matching measured preflight: a 3 GiB cap for this checkpoint's
new retained evidence, at least 8 GiB free, 25% projected growth plus 128 MiB reserve,
runtime guards and at most four single-thread workers. Rejected gates require an
explicit budget/design revision. All previous checkouts/studies remain preserved.
Detailed raw inputs/models/predictions/private indexes stay local. Publish only
code, frozen settings, documentation, figures and compact aggregate results.
