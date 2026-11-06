# Prespecified fresh shared-GMM neural validation

The source-frozen study compares the new opt-in shared-GMM adapter with unchanged
HDBSCAN and full GMM on identical separate-encoder coordinates for each neural
seed (0, 101, 202). Each branch trains once per sampled input; clustering heads
reuse exact coordinates. Full GMM remains benchmark-only for this neural adapter.

Main data: 520 fresh independent observations of datasets, each with N=64/256
matched individuals, four morphology and eight ecology traits. For each N, there
are 100 independent-Gaussian and 100 correlated-Gaussian controls, plus ten each
for morphology-only, ecology-only, complementary and redundant four-group signals,
unequal-spread complementary groups and noisy moons. The Gaussian laws are
single-population controls; unequal spread and moons test limitations. The new
unequal-spread law multiplies group residual SD by 0.5/1/1.5/2 while retaining the
complementary means and independent feature noise SD=0.15. Original simulators
remain unchanged, with an independently implemented reference for the new law.

The 13 point evaluations per dataset include three neural seeds times three
clustering heads, the same three heads on existing per-view PCA coordinates, and
a standardized raw-trait shared-GMM reference. GMMs use BIC, K=1..4, hard minimum
size five, ten starts, 500 iterations, tol=0.001, reg_covar=1e-6 and seed 71323.
No absolute likelihood/BIC comparison crosses coordinate spaces or dimensions.

Ten matched 80% subsamples attach to the first dataset in each morphology-only,
complementary, independent-Gaussian and unequal-spread cell (eight datasets).
All heads use the same fitted coordinates and sample IDs; neural seeds remain
separate repeated measurements. Twelve archived numerical replays verify legacy
encoders/HDBSCAN and the completed frozen-coordinate GMM results. Replays do not
enter fresh estimates. The main budget is 7,516 clustering evaluations and 3,624
neural branch trainings; a separate 16-dataset preflight uses 568 evaluations and
336 trainings. Total: 8,084 evaluations, 3,960 trainings and 1,980,000 Adam updates.

Freeze seeds, settings, source and endpoints before preflight. Audit every actual
optimizer update, raw identity joins, branch transforms, density hierarchy,
Gaussian likelihood/probabilities, candidate eligibility/BIC, exact replays and
subsampling denominators independently. Retain failed/unattempted searches and
partial numerical evidence without replacement, and preserve failed guards.

Report 108 primary rates for the six signal/control laws across N, seeds and
heads, and 72 paired shared-vs-HDBSCAN/full comparisons. Include exact marginal
and simultaneous intervals, failure bounds, correct K, ARI, coverage, constraint
failures and descriptive stability. Seeds/heads/draws are repeated measurements;
only generating datasets are independent replicates. Ten signal datasets/cell
and one stability dataset/cell support bounded rather than general validation.

Gaussian K>1 selections are spurious-split diagnostics, not formal significance
rejections or calibrated species-test type-I error. Membership probabilities are
conditional on fitted parameters. No null calibration, winner selection, default
change, empirical fitting, empirical-gate change or author contact occurs.

A measured full-settings preflight must project main serial execution plus audit
below two hours with 25% allowance and no parallel credit. Require 4 GiB retained
new evidence, an additional 128 MiB storage reserve in projection, at least 8 GiB
free and at most two single-thread research workers. Detailed raw/fit/audit
artifacts remain local; only code/docs/invented fixtures/figures/cell aggregates
are published. Validate Python 3.11/3.12 tests, installed packages and exact-head
master CI, then stop regardless of recovery outcome.
