# Shared-covariance grouped morphology–ecology calibration

```bash
python -m pip install '.[clustering]'
super-deli calibrate-grouped-integrative /path/to/dataset.json \
  --calibration-config examples/shared-grouped-calibration/calibration.json \
  --output-dir /path/to/new-calibration
```

Python 3.11/3.12 are supported. Fix `model.covariance_type="tied"` before
examining clustering or p-values. Components have separate means/weights and
share one full covariance matrix. The clustering default remains `"full"`.

This workflow reuses the [schema-2 conditional-PSD grouped
null](grouped-covariance-boundaries.md), with the same estimator, bounded
optimization recipe, convergence guards, raw simulator and seed namespace.
It enables the prespecified shared-family clustering statistic only in schema 2.
Legacy schema 1 retains its full-family behavior and rejects shared covariance.
The existing matched joint genomic–phenotypic shared-family workflow remains
separate. Shared covariance here describes score-space GMM components; it does
not equate the raw between-group and within-specimen covariance matrices. Genomic/shape/count inputs, missingness and informative sampling sizes
are outside this grouped null.

## Input, hierarchy and fixed unequal counts

Use complete continuous specimen morphology explicitly linked to populations or
localities, with exactly one directly measured ecological vector per group.
Every declared group must have morphology and ecology. Preserve specimen,
observation, feature and group IDs and source row orders. Ecology is never
copied onto specimens as independent measurements.

Assume independent Gaussian group effects and iid multivariate Gaussian
morphology residuals independent of those effects. Counts are fixed within a
test and non-informative for the raw biological features. Within-group residual
covariance is shared across groups. There is no spatial/kinship model, repeated
measure model, heterogeneous residual law or separately identifiable ecology
measurement-error component. Declarations assert these assumptions; the software
does not infer that empirical samples satisfy them.

Fit the existing raw marginal maximum likelihood before PCA. The conditional
parameterization fits morphology on group ecology, with positive-semidefinite
conditional between-group morphology covariance, including zero/reduced rank.
Ecology and within-morphology covariance must remain positive definite. Unequal
counts enter the likelihood through group-mean covariance and within-group
scatter. Rank refers to conditional covariance, not species or clusters.

Retain the versioned four-start bounded factor/continuation and projected-polish
procedure, constrained-optimality checks and accepted/rejected stage diagnostics.
No variance floor, ridge, predictor removal, adaptive restart or post-hoc solver
change is introduced. Best accepted numerical likelihood and stationarity do
not certify a global optimum. Non-identifiable, singular or unresolved fits
retain failures and withhold inference.

## Raw-data bootstrap and statistic

Fit the raw null once per observation and hold its parameters fixed for 99
independent draws. Preserve all observed specimen/group counts and identities.
Generate new raw specimen morphology and one ecology row per group. Refit
CSV ingestion, tabular preprocessing, specimen morphology view PCA, explicit
mean aggregation to groups, ecology view PCA, weight normalization/fusion,
joint PCA and every prespecified shared-GMM K candidate. The model settings and
model seed are identical for observed and null point analyses; upstream scores
are never frozen across draws. Raw null parameters need not be refitted within
a draw because they do not enter the chosen point statistic.

Use `T = BIC(K=1) - min_eligible BIC(K=1..Kmax)`. Shared-family parameter count
in d score dimensions is `K*d + d*(d+1)/2 + K-1`. Dense exports repeat the common
covariance for each component; BIC counts it once. Include ties in the upper
tail: `p=(1+count(T_draw >= T_observed))/(B+1)`. With B=99 the minimum is 0.01;
alpha=0.05 uses p<=alpha. T=0/p=1 is expected when K=1 wins.

P-values are approximate fitted-model departure measures. Covariance-boundary
fits are nonregular. Full pipeline refits and a plus-one calculation supply
neither exact exchangeable-rank inference nor a general/uniform size guarantee.
Conditional Clopper–Pearson tail intervals measure draw variation given observed
T, fitted parameters and fixed counts; they exclude null-estimation, parameter/model
and sampling-assumption uncertainty. Failure-outcome bounds are not confidence
intervals. Any failed requested draw withholds the primary p-value, decision
and interval; failed points leave draws unattempted. No draws are replaced.

Stability runs separately. There is no cluster-specific, selected-K or
species-boundary significance. Select the covariance family, PCs, view weights
and features in advance; post-hoc selection between configurations is outside
this calibration. Heavy tails may reject without discrete biological groups.

## Frozen fresh validation

The [study](../examples/shared-grouped-calibration/study.json) fixes seven
scenarios: independent Gaussian, correlated Gaussian, complementary four groups,
heavy-tailed t3, zero conditional covariance, rank-two conditional covariance
and weak full-rank conditional covariance. All generating laws and the schema-2
solver are unchanged. The heavy-tail scenario is a misspecification stress;
the others supply model-law null checks or the constructed alternative.

Use 25 fresh observations/case, 64 groups, independent uniform 1–4 morphology
specimens/group, four features/view, two PCs/view and jointly, equal weights,
K=1..4, minimum size 5, 10 EM starts, B=99 and alpha=0.05. Master seed 20261102
and model seed 71323 are fixed before fresh data. There are **175 observations
and 17,500 planned pipeline fits**, with independent fitted nulls, no shared draw
bank, tuning, replacement fits or adaptive stopping. Run at most four one-thread
case jobs. A separate full-settings preflight uses a different seed.

```bash
super-deli benchmark-grouped-calibration examples/shared-grouped-calibration/study.json \
  --case independent_gaussian --output-dir /path/to/new-study/independent_gaussian
# Repeat for the other six declared cases.
super-deli analyze-grouped-calibration /path/to/new-study \
  --output-dir /path/to/new-audit
```

The existing no-refit audit regenerates observations/raw draws and verifies
counts/identities, dense raw likelihoods, covariance-cone optimality, ranks and
sampling factors, fusion/projection, selected mixture likelihood/BIC/membership,
shared-matrix equality, K selection, seeds, tails, intervals and denominators.
View PCA and nonselected likelihoods are retained/checksummed, not independently
refitted. Outer failures retain attempt/error accounting rather than a complete
numerical audit. Rates across independent observations receive pointwise Wilson
intervals and all-requested withheld-outcome bounds, separate from per-test
Monte Carlo intervals. Small scenario sizes do not establish precise 5% control.
Historical studies use different observations; family-effect comparisons would
be unpaired. A small installed CLI smoke is provided separately for CI.

See the [completed fresh study](shared-grouped-calibration-results.md) for
audited outcomes, availability, covariance diagnostics and limitations.
