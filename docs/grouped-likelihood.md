# Opt-in grouped AIC selection and full-pipeline likelihood calibration

This checkpoint adds `selection_criterion="aic"` for full-covariance grouped
morphology/ecology clustering with group-mean morphology PCA. **BIC remains the
default**; its configuration bytes, selection behavior and historical p-values
are preserved. No covariance, PCA, solver, ten-start or size-filter default changes.
The previous [diagnosis](full-grouped-diagnostics-results.md) motivates these
separate component-selection and departure-detection changes.

**Research status:** the frozen validation improved alternative detection and
recovery, but boundary Gaussian nulls rejected in 12–16% of observations at nominal
5%, and AIC often selected multiple components without global rejection. Boundary
size and point overpartitioning remain unresolved; see the [results](grouped-likelihood-results.md).

## Selection and test statistic

For d joint PCs and K full-covariance components, count
`q=(K-1)+K*d+K*d*(d+1)/2`. AIC selects the smallest eligible
`AIC=-2*ell_K+2*q`, resolving exact ties toward smaller K. Existing BIC uses
`-2*ell_K+q*log(N)`. Both include K=1. AIC is a component-selection heuristic;
mixture nonregularity and finite-sample overfitting prevent a general guarantee
of recovering the generating K. Its conditional memberships and whole-group
bootstrap stability do not become species probabilities or significance levels.
[Akaike's original paper](https://doi.org/10.1109/TAC.1974.1100705) defines the criterion.

The new `calibrate-grouped-likelihood` workflow uses the global statistic
`T=2*(max_eligible ell_K - ell_1)` over the complete prespecified K search.
K=1 ensures T>=0. It uses fitted, covariance-regularized component densities and
the unchanged convergence/minimum-hard-size rules, rather than claiming the
global unconstrained likelihood maximum. The statistic is independent of which K
AIC selects. The old BIC-improvement calibration remains a separate workflow.
Changing criteria, statistics, families, dimensions or weights after inspecting
results is not calibrated. No chi-squared reference is used: Gaussian-mixture
likelihood tests violate the usual regularity conditions.
[McLachlan's bootstrap study](https://academic.oup.com/jrsssc/article/36/3/318/6985676)
is methodological motivation, not a validity theorem for this integrative pipeline.

AIC retains the existing ten-start optimizer's best likelihood fit for each K,
then applies the existing component-size screen. It does not choose the best
admissible start from the restart pool. More starts can still reduce eligibility;
that known behavior and the earlier separate numerical issue are unchanged.

## Raw null and complete refits

A versioned `max_eligible_likelihood_gain_v1` declaration requires the existing
schema-2 conditional-PSD hierarchical raw Gaussian null: complete continuous
specimen morphology with explicit population/locality membership, one directly
measured ecology row/group, independent groups, iid within-morphology Gaussian
residuals and fixed non-informative unequal counts. Between-group conditional
morphology covariance may be PSD; ecology and within-morphology covariance must
be positive definite. The estimator, bounded optimization, convergence guards
and sampler are reused unchanged.

Fit raw nuisance parameters once per observed dataset and hold them and the
design fixed within that test. Each of 99 independent draws regenerates raw
morphology/ecology and refits ingestion, equally weighted raw group means,
standardization, both view PCAs, aggregation, fusion, joint PCA and every GMM K.
No ecology is broadcast onto specimen rows as independent evidence. Group means
with fewer specimens remain noisier despite equal PCA influence.

Use `p=(1+count(T_draw>=T_observed))/100`, including ties, and alpha=0.05.
P-values are approximate plug-in Gaussian-model departure tests. Raw estimation,
covariance boundaries, finite optimization and misspecification preclude general
or uniform size guarantees. A rejection does not establish discrete species,
selected K or membership confidence. Clines and heavy tails can cause valid
model-departure rejections without discrete generating groups. Non-rejection
does not validate Gaussianity or a biological delimitation.

Any failed draw withholds the primary p-value, decision and tail interval.
Failed observed points leave draws unattempted; raw-null failures also withhold
inference. All errors, attempts and missing-outcome bounds remain, without
replacement draws, observations or selective retries. Conditional Clopper–Pearson
tail intervals measure draw variation given fitted parameters/design, excluding
nuisance-estimation, numerical and misspecification uncertainty. Across-observation
Wilson intervals are pointwise and unadjusted. Neither is species confidence.
Stability remains separate; calibration requires zero stability replicates.
Other significance workflows explicitly reject the new selection option.

## Frozen paired validation and audit

Master seed 20261114 fixes 25 fresh observations in each of ten scenarios:
independent/correlated Gaussian nulls, complementary four groups, heavy tails,
zero/rank-two/weak conditional-covariance Gaussian nulls, unequal-variance four
groups, continuous clines and complementary four groups at N=256. The other nine
scenarios use N=64. Keep four traits/view, two PCs/view, four joint PCs, equal
weights, full covariance, K=1..4, minimum size 5, 10 starts, model seed 71323,
99 draws and alpha=0.05. At most four one-thread jobs perform **25,000 unique
point-pipeline fits**. Twenty-five separately archived point replays verify legacy
candidate fits, scores, BIC selections and selected densities before fresh data.
Seed 20261115 supplies a separate developmental smoke, without main-study replacements.

Each fitted candidate pool supplies AIC/BIC component choices and likelihood/BIC
statistics on the same scores. Each test's raw null draws are also shared, so
paired policy comparisons require no duplicate fitting. The likelihood-gain test
is primary; BIC is a prespecified comparator. Taking the smaller p-value is not
calibrated. Generating labels are used only after fitting, never to initialize,
select K or choose settings. Report point overpartitioning under null/stress laws
separately from rejection, recovery and calibrated power.

The AIC option retains `gmm_candidates.npz`, containing scores and every fitted
K's weights, means, full covariances, memberships and labels, including
size-ineligible candidates. Hashes and complete attempt logs accompany it.
The no-refit auditor independently reconstructs every candidate density,
likelihood, membership and AIC/BIC count, plus raw dense likelihood/constrained
optimality, all PCA moments/eigen-equations/projections, fusion, counts/identities,
seeds and tail accounting. Convergence does not certify global optimization.
Outer failures keep partial evidence without a full numerical re-audit.

Full frozen evidence remains external; [portable results](grouped-likelihood-results.md)
include paired rates, K counts, ARI and Monte Carlo uncertainty, source/runtime
fingerprints and figures. Existing checkouts, datasets, p-values and evidence are
preserved. No new default is promoted, and Python 3.11/3.12 remain supported.

```bash
super-deli cluster-integrative /path/to/dataset.json \
  --model-config examples/grouped-likelihood/model.json \
  --output-dir /path/to/new-point-run
# For separate whole-group stability, use stability.json (100 requested replicates).
super-deli calibrate-grouped-likelihood /path/to/dataset.json \
  --calibration-config examples/grouped-likelihood/calibration.json \
  --output-dir /path/to/new-calibration
super-deli benchmark-grouped-likelihood examples/grouped-likelihood/study.json \
  --case independent_gaussian --output-dir /path/to/new-study/independent_gaussian
# Run the other nine declared cases into siblings, at most four one-thread jobs.
super-deli analyze-grouped-likelihood /path/to/new-study \
  --output-dir /path/to/new-audit
```
