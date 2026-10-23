# Full-pipeline calibration with group-mean morphology PCA

The opt-in `pca_fit="target_means"` policy is supported for **schema-2 grouped
morphology–ecology calibration with prespecified shared covariance**. Source-row
PCA, two joint PCs and full covariance remain defaults. Legacy schema 1 and all
other target-mean significance workflows remain disabled. Historical settings,
configuration serialization and p-values remain unchanged.

## Raw null and fixed sampling design

Use complete continuous specimen morphology linked explicitly to populations or
localities, plus one directly measured ecology row per group. Assume independent
Gaussian groups, iid Gaussian morphology residuals independent of the group
effects, and fixed non-informative specimen counts. The unchanged
[conditional-PSD raw estimator](grouped-covariance-boundaries.md) fits joint group
means/covariance before PCA by marginal maximum likelihood, allowing zero or
reduced-rank conditional between-group morphology covariance. Ecology and
within-morphology covariance must be positive definite. Retain the existing bounded
optimization/continuation policy and constrained-optimality checks. There is no
new variance floor, ridge, adaptive restart or latent-error-free mean estimate.

Counts and every source identity/link are fixed in all draws. Groups with fewer
specimens retain noisier raw means. Ecology is never copied to specimens as
independent evidence. One ecology row/group cannot separately identify ecology
measurement error. Informative sampling, related/spatial groups, heterogeneous
within-group residual covariance, missingness, counts, genomics and shapes are
outside this null.

## Complete refits and interpretation

Fit one raw null to the observed inputs, then hold those nuisance parameters and
the observed sampling design fixed within its test. Each independent draw generates
new raw specimen morphology and group ecology. Observed and null pipelines use the
same prespecified configuration and model seed:

1. Ingest raw features and resolve explicit group membership.
2. Compute raw morphology means using every member; fit standardization and fixed
   morphology PCA with one equally weighted row per group. Fit ecology PCA on its
   group rows. Project specimen rows using each draw's newly fitted parameters.
3. Aggregate projected morphology by group, normalize the view blocks, fuse them
   with fixed weights and refit joint PCA.
4. Search the entire configured shared-covariance GMM K grid and apply the existing
   convergence/distinct-unit/minimum-size rules. Count shared covariance once in BIC.

The global statistic is `T = BIC(K=1) - min_eligible BIC(K=1..Kmax)`.
With 99 draws, `p = (1 + number(T_draw >= T_observed))/100`; ties enter the upper
tail, minimum p is 0.01, and rejection uses p<=0.05. This is an **approximate
plug-in model-departure test**. Fitted nuisance parameters remove exact-rank
exchangeability; covariance-boundary fits are **nonregular**, with no uniform or
general size guarantee. Rejection neither identifies species nor establishes
selected K, boundaries or membership confidence. Non-rejection does not establish
Gaussianity. No selection across PCA policies/dimensions, covariance families,
weights or features is calibrated; fix them before looking at outcomes.

Conditional Clopper–Pearson tail intervals quantify draw variation given observed
T, fitted parameters and counts. They exclude nuisance-estimation and model
uncertainty. Benchmark rejection rates use independent observed datasets with
pointwise Wilson intervals, unadjusted across the seven scenarios. These are
separate uncertainty measures. An atom at T=0/p=1 and a zero plug-in Monte Carlo SE
do not imply absence of uncertainty. Stability is a separate workflow.

Failed points leave draws unattempted. Failed null draws withhold the primary
p-value, decision and conditional interval. Preserve all errors/attempt ledgers and
failed-outcome bounds; never replace, repair or selectively retry datasets/draws.
Bounds are not confidence intervals. Unresolved raw fits also withhold inference.

## Commands and frozen validation

```bash
super-deli calibrate-grouped-integrative /path/to/dataset.json \
  --calibration-config examples/groupmean-calibration/calibration.json \
  --output-dir /path/to/new-calibration

super-deli benchmark-grouped-calibration examples/groupmean-calibration/study.json \
  --case independent_gaussian --output-dir /path/to/new-study/independent_gaussian
# Run the other six declared cases into sibling directories, at most four at once.
super-deli analyze-grouped-calibration /path/to/new-study \
  --output-dir /path/to/new-study-audit
```

The study freezes master seed **20261108**, model seed **71323**, 25 observations
for each of independent/correlated Gaussian, complementary four-group, heavy-tailed,
zero conditional covariance, rank-two and weak full-rank conditional covariance
laws. All have 64 groups, independent 1–4 morphology specimens/group, one ecology
row/group and four continuous features/view. Retain two PCs/view, **four joint
PCs**, equal view weights, shared covariance, K=1..4, minimum cluster size 5 and
10 EM starts. Use 99 draws/test, alpha=0.05 and unchanged solver budgets:
**175 observations and 17,500 planned point-pipeline fits**, at most four one-thread
jobs. Preflight uses a distinct seed 20261109 and supplies no main-study replacements.
Settings are fixed before fresh generation; generating labels never enter fitting.

The unchanged simulator supplies the declared laws. Five Gaussian cases check the
raw null; complementary groups evaluate constructed power, and heavy tails assess
misspecification sensitivity. Twenty-five observations per scenario provide coarse
rate uncertainty and no general 5% guarantee. New observations differ from historical
studies, so comparisons are unpaired and do not isolate the PCA-policy effect.

The read-only audit regenerates observations and all raw draws, verifies archives,
identities/counts, dense raw likelihood/constrained convergence, seed streams,
selection/BIC/membership and tails. For every successful observed/null pipeline it
also independently reconstructs raw group means, feature means/scales, PCA
eigen-equations, projected rows and aggregated profiles for both views. It does not
refit models; retained nonselected GMM likelihoods are not independently refitted,
and convergence does not certify a global optimum. Previous #55 numerical failures
are preserved; their runtime issue is outside this checkpoint.
