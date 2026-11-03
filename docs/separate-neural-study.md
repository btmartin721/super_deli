# Prespecified separate-encoder comparison

The optional `benchmark-separate-neural` lifecycle exposes `bind`, `freeze`,
`preflight`, `audit-preflight`, `gate`, `main`, `audit-main` and `report`, with
required `--archive-root` and `--output-dir`. The source root is the completed
view-weight study; twelve equal-weight archived points are checksum bound.
Keep the approved plan as `approved-plan.md` in a new local output root. This
research benchmark is separate from normal clustering use.

The frozen design has 80 fresh main datasets: five each across eight laws and
N=64/256. The first six laws are morphology-only signal, ecology-only signal,
complementary four groups, redundant four groups, independent Gaussian and
correlated Gaussian. Discordant partitions and noisy moons are diagnostic
stress; discordance has no imposed common species truth. All raw laws retain
four morphology/eight ecology traits and 0.15 measurement noise. Phase/data/noise/
row-order/subsampling RNG streams are independent and frozen before generation.
Generating labels and latent coordinates enter evaluation only after fitting.

Seventeen point arms retain all three neural seeds (0/101/202): separate encoders,
unchanged equal-weight joint-two encoders and benchmark-only joint-four encoders
(nine arms); six unchanged single-view neural references; one per-view two-PC
concatenation/HDBSCAN control; and the unchanged integrative PCA/GMM baseline.
The joint-four control uses d/16/8/4 and 4/8/16/d networks with the same optimizer,
input balance and 500 updates. The PCA control standardizes its two score axes
per view and retains all four. Both controls are benchmark-only, without new
public model adapters. Dimension matching does not equalize parameters/compute.

The 1,360 main point workflows receive 108 paired native subsampling refits on
the first morphology-only/complementary observation at each sample size, plus
twelve exact legacy point replays: **1,480 main workflows**. The separate
sixteen-observation preflight has 272 points plus 108 refits: **380 workflows**.
These correspond to 1,596 main and 432 preflight neural trainings, or 1,014,000
optimizer updates. Density fits and neural trainings overlap workflow counts;
seeds/draws/controls are repeated measurements, not additional datasets.

Read-only independent NumPy gradient/moment/Adam and loss checks observe actual
primary optimization under unchanged strict tolerances. Audits verify raw laws,
joins, scaling, branch/four-coordinate transforms, rank, PCA, density hierarchy,
GMM densities/BIC/membership and native support denominators. Separate point
branches must exactly reproduce their corresponding standalone weights, losses
and scores (480 planned main comparisons); archived joint-two points have twelve
exact initial/final/loss/score/label/strength replays. No second optimization is
run merely for the numerical audit. Partial/failed fits remain explicit without
replacement, K=1 fallback or favorable reassignment.

Reports retain all 272 cell/arm groups and 36 illustrative native-support
summaries. The first six laws define 108 primary rates and 72 within-seed paired
comparisons against joint-two, with marginal/simultaneous uncertainty and failure
bounds. Four-group recovery requires K=4 and full ARI >=0.8, with noise units as
distinct unresolved singletons. Gaussian splitting is K>1, not a calibrated
type-I error. Moons recovery and agreement with each discordant partition remain
secondary. Report coverage, full/assigned agreement, reconstruction, linear
axis retention, rank, neural parameters, training counts and workflow cost.
Select no winning seed, weight, family or new default; compare no absolute BIC
across score spaces. Five independent datasets/cell and three subsamples on four
observations per arm support a descriptive pilot, not general superiority or
species support.

Before main execution, the independently audited preflight must pass a measured
**2 GiB** local-evidence cap, **8 GiB** free reserve, at most **two one-thread
workers**, and projected main execution plus audit below **one hour**, with 25%
allowance and no parallel speed credit. Projection uses per-sample-size/arm
maxima of observed training groups, refits, workflow/source overhead and retained
bytes; both branch trainings are counted explicitly. Logs/test archives/verification staging
count toward evidence; development and CI time are outside the runtime gate.
Numerical/resource guard failures remain retained and stop dependent work for an
explicit amendment. No setting, tolerance, seed or study size changes after fresh
outcomes; verified lossless compaction may replace only new generated scratch.

Engineering checks cover pure schemas/lazy help, joins, constants/redundancy,
exact branch/native parity, frozen projection/reconstruction, failure gates and
package allowlists on Python 3.11/3.12. Only code/docs/figures/small aggregates are
published. Detailed inputs, networks and diagnostics remain local. No empirical
fitting, significance extension, missing/grouped/unmatched neural adapter,
genomic/shape neural adapter, anatomical learning or author contact is included.
