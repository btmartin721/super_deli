# Prespecified initial neural calibration pilot

The checkpoint fixes forty fresh datasets at N=64: ten each from independent and
correlated Gaussian laws, complementary balanced four-group signal, and heavy-tail
stress. Each contains four morphology and eight ecology traits, with independent
feature noise SD 0.15. The first three laws reuse the validated matched-view
simulator. Heavy-tail stress multiplies each independent two-coordinate latent
residual vector by `sqrt(3/chi_square_3)` before applying the same feature maps and
noise. An independent reference implementation checks every raw dataset.

Each observation gets its own fitted joint raw-Gaussian null and 99 fresh raw
draws. Neural seed 0 is prespecified as the existing default, with mixture seed
71323; there is no initialization selection. Both unchanged encoders retain two
coordinates, concatenated into four. The benchmark-only PCA reference retains
two standardized PCs per view without an additional reduction. Both heads use
shared covariance, BIC, K=1..4, minimum hard size five, ten starts, 500 EM
iterations, tolerance 0.001 and regularization 1e-6. Raw nulls receive no ridge.

Both arms use exactly the same observations, raw null estimates, null draw seeds
and generated values. Every pipeline refits ingestion and transformations. Heads
are repeated measurements of a dataset; draws are repeated conditional
simulations, not independent study observations. Four exact prior point replays
are separate from inference. No stability bootstrap, empirical analysis,
architecture/seed search, new default or author contact occurs.

The main budget is 4,000 raw inputs, 8,000 candidate searches, plus eight searches
for four paired point replays: 8,008 searches and 8,008 branch trainings with
4,004,000 actual optimizer updates. A separate full-settings preflight uses one
independent dataset per law, 400 raw inputs, 800 searches and 800 branch trainings
with 400,000 updates. Invented engineering/installation fixtures are separate.

Freeze source, settings and RNG roots before scientific generation. Main roots
are 110017/110023/110039/110051, with independent preflight roots
110059/110063/110069/110083. Preserve numerical, source and resource guard
failures, retain failed draws without replacements and withhold incomplete
p-values. Every actual optimizer update receives the existing independent NumPy
gradient/Adam checks. Independently audit raw moments/likelihood, generated
values and identities, branch/PCA transforms, dense candidate densities,
membership, BIC/eligibility, exact replays, seeds, tails and denominators.

Primary reporting contains sixteen rates: two arms times four laws times global
rejection and ordinary K>1 selection. Include exact marginal 95% intervals,
Bonferroni simultaneous intervals and failure bounds. Correct-K/ARI and the joint
rejection/recovery event are secondary signal outcomes. Eight paired event
comparisons retain missing-pair accounting. Heavy-tail rejection measures
Gaussian-model departure; it is not species or clustering type-I error. Ten
datasets per law provide an initial pilot with wide uncertainty, not a general
size or power guarantee. BIC and likelihood are compared within one score space.

The independently audited full-settings preflight gates main execution. Require
projected serial execution plus audit below four hours including a 25% allowance,
with no parallel speed credit. At most two single-thread workers may run. Retained
new research and verification evidence must fit within 4 GiB, with measured
staging/copy costs, a 128 MiB projection reserve and at least 8 GiB free. Tooling
dependencies/caches are separate from the evidence quota; all disk use reduces
the reserve. Any failed guard is retained and requires an amendment.

Only code, documentation, invented examples, figures and cell aggregates are
published. Full inputs, weights and per-fit numerical evidence stay local.
Python 3.11/3.12 tests, package checks and exact published-master CI must pass
before closing the checkpoint.
