# Nonlinear measurements and supervised residual networks: frozen protocol

This bounded simulation pilot evaluates reference-class assignment separately
from unsupervised discovery. It adds no p-values, significance/stability draws,
empirical analysis, new clustering algorithm or automatic model promotion.
Existing models, defaults, serialization, empirical gates and historical p-values
retain their behavior. The residual classifier receives reference labels and the
reference class count; that information is unavailable to the discovery arms.

## Measurements and cohorts

Use the existing four morphology/eight ecology loading matrices, unit-residual
two-dimensional latent variables per view and feature noise SD .15. Four balanced
reference classes shift morphology axis zero and ecology axis one by their two
corner signs. Standard shifts are +/-3; weak shifts are +/-1.5. Morphology and
ecology row orders are shuffled independently and matched by explicit IDs.

Let base observed traits be `m,e`, and independent size noise be `s~N(0,.5²)`.
Allometry maps morphology to `exp(.35*m + s*a)`, with `a=(.8,1,1.2,1.4)`;
saturation maps ecology to `20+5*tanh(e/2)`. Noise is already present in base
traits before these maps. These are illustrative mechanisms, not empirically
estimated measurement or speciation models. They may alter observable information
as well as geometry; neural superiority is not assumed.

The eight laws are weak linear groups; standard linear groups; allometry only;
saturation only; combined allometry/saturation; weak groups with confounded size;
one Gaussian population; and the existing continuous shared gradient. In the
confounded law, size is associated with the morphology class axis by +/-0.8 in
training/IID observations, and ecology additionally receives `.4*s*b`, with
`b=(1,-1,1,-1,1,-1,1,-1)`. Shifted evaluation removes that size/class association.

Main has five independent observations for each law at N=64/256: 80 datasets.
Separate preflight has one observation/cell: sixteen datasets. Each observation
has 2,048 independent evaluation units. The sixty labeled main observations
also have 512 unfamiliar-class units with underlying group-axis means (+6,+6).
The ten confounded main observations also have 2,048 shifted units. Cohorts have
disjoint IDs and innovations. Evaluation never changes preprocessing/training.
No labels are invented for the Gaussian or gradient controls.

## Arms and budgets

On labeled observations fit the supervised residual network at seeds 0/101/202,
multinomial logistic regression, existing per-view PCA/shared-GMM and existing
separate-encoder/shared-GMM at all three seeds. Logistic regression uses the same
training-standardized balanced raw features, L2 C=1, lbfgs, 2,000 maximum updates
and tolerance 1e-9. The four discovery arms alone run on controls. Existing GMM
settings remain shared covariance, BIC K=1..4, minimum size five, ten starts,
mixture seed 71323, max_iter=500, tolerance .001 and regularization 1e-6.

One fixed training-label permutation on the first standard-linear observation at
each N adds eight negative-control fits/phase. Twelve exact main point replays
use the first three archived weak observations at each N, with original PCA and
neural seed-zero results. Audited lossless extracts bind the original power-study
arrays and source-completion/CI provenance. Replays do not enter fresh estimates.

Main is 580 workflows; separate preflight is 120. Combined maxima are 816 neural
trainings/408,000 optimizer updates, 76 logistic fits, 396 GMM searches and 1,584
candidate component fits/15,840 starts. No new calibration or stability draw is
requested. Engineering fixture fits are recorded separately.

## Evaluation and boundaries

Known-class supervised metrics are balanced accuracy, confusion, log loss and
multiclass Brier score, alongside labels used, parameter counts, runtime, seed
sensitivity and nuisance-shift results. Unfamiliar-class confidence is descriptive;
no unknown-class rejection rule, detection rate or species interpretation is defined.

Discovery reports selected K, training/frozen-evaluation ARI, K=4 plus ARI>=.8
recovery, Gaussian splitting and failures. Gradient partitions remain descriptive.
Compare no absolute BIC between different representation spaces.

Seed zero is primary; other seeds are sensitivity measurements. Forty primary
dataset-level rates comprise twelve residual-versus-logistic gains of at least
five accuracy percentage points; twenty-four four-group recovery rates; and four
Gaussian split rates. Retain exact marginal and separate forty-rate simultaneous
95% intervals, requested denominators and unavailable-outcome bounds. Five
datasets/cell support a descriptive pilot; seeds and evaluation specimens do not
increase the count of independent dataset replicates. No winning seed or default
is selected after outcomes.

## Lifecycle, resources and audit

`benchmark-residual-nonlinear` provides bind/freeze/run/audit/gate/report actions
with explicit source/output roots. Scientific settings, code, plan, numerical
runtime and seed paths are frozen before generation. The source is the completed
weak-signal archive, whose extracts retain the completed power-study point fits.
The engineering fixture has a separate marker and seed scope.

Independent NumPy equations check every actual neural gradient/Adam update,
source joins, nonlinear laws, training-only preprocessing, frozen predictions,
class ordering, densities/BIC, metrics and intervals. Preserve failures without
replacement and withhold affected results. Numerical/source/resource guard
failure stops dependent work; changes require an explicit amendment.

Require an independently audited full-settings preflight, at most two single-thread
workers, a two-GiB local-evidence cap, eight-GiB free reserve, twenty separate
serial preflight minutes and ninety serial main/audit minutes. Projections include
25% allowance, staging counted twice, a 128-MiB reserve and no parallel speed credit.
Code/docs/figures/small summaries are public; individual research measurements,
labels, predictions, network weights and detailed audit traces stay local.
Verify Python 3.11/3.12 tests, installed packages and all six published-head CI jobs.
Stop at this checkpoint regardless of relative performance. No author contact.
