# Archival weak-signal recovery diagnosis

`diagnose-weak-signal` is a separately versioned, opt-in research diagnostic. It
uses completed, source-bound synthetic observations and saved coordinates. It
adds no production model, species test, empirical cohort or changed default.

The real schema-1 design binds the completed power study at `cf6082d`, its six
CI receipts, original source freeze, approved runtime repair and full selected
unit inventories. Exactly 80 original fresh observations are selected: ten per
six four-group cells (N=64/256, center multipliers 0.5/1/1.5), and ten independent
Gaussian controls per N. Labels are evaluation or explicitly supervised
initialization evidence; they never choose an unsupervised configuration.

## Fixed analyses

Known generating means/covariance provide three reference classifiers: stored
four-coordinate latent variables, twelve raw traits and four frozen PCA scores.
The raw law includes the fixed linear loadings and independent noise SD 0.15.
PCA means/covariance are pushed through its audited affine transformation.
Pointwise equal marginal priors are used without enforcing known group counts.
Report accuracy, adjusted Rand agreement, ambiguity and model-implied expected
classification error. Treat fitted PCA maps as fixed for this last calculation.
These references are not finite-sample ARI ceilings or species probabilities.

Trace in-sample linear recovery, ranks and within/between scatter through raw,
latent, PCA and frozen neural representations. A nonlinear encoding is not
assumed to have a Gaussian pushforward. Geometric or linear-recovery changes
alone do not establish loss of all recoverable information.

Independently audit all 160 archived heads, including every K=1..4 candidate,
dense likelihood, BIC, convergence, size eligibility and membership. Keep native
selection and the stored K=4 partition separate. Compare likelihood gains and
BIC costs only within the same coordinate space.

Replay exactly twenty ordinary shared-GMM searches: both representations on
weak-signal replicates zero through four at both N. Require exact scientific
parity with archived candidates/arrays. Fit 120 separate, generating-label
initialized K=4 diagnostics: two spaces on each four-group observation, one
start each. Retain the production regularizer, seed, tolerance and iteration cap;
apply the size rule descriptively. These fits never enter production selection
or historical inference. Controls have no four-group truth initialization.

## Lifecycle and budgets

Run `python -m super_deli diagnose-weak-signal --help`. Each action requires
`--archive-root` pointing directly to the completed power-study directory and a
separate `--output-dir` containing the approved plan as `approved-plan.md`.
Use `examples/weak-signal/study.json` and execute `bind`, `freeze`,
`run --phase preflight`, `audit --phase preflight`, `gate --staging-bytes BYTES`,
`run --phase main`, `audit --phase main`, then `report`.

The separate preflight uses the eight replicate-zero observations, 24 reference
evaluations, four replay searches and twelve truth starts: at most 28 component
fits/172 EM starts. Main uses 240 reference evaluations, twenty replay searches
and 120 truth starts: at most 200 component fits/920 EM starts. There are no new
scientific datasets, neural trainings, null draws or stability subsamples.

Freeze source, settings, input hashes and numerical versions before fitting.
Use at most two one-thread workers. Require a measured 25% projection allowance,
64-MiB reserve, one-GiB retained-evidence cap including staging/verification
copies, and eight-GiB free space. Main execution/audit must project below thirty
serial minutes; separate preflight below ten. Tool environments are excluded
from the evidence quota but count against physical free space.

Retain failed diagnostics without replacement and withhold their endpoints.
Source, numeric, exact-replay and resource guard failures retain a STOP and halt
dependent work. Sealed units are verified on continuation; interrupted work is
retained without a replacement attempt. A failed resource gate requires an
explicit amendment or approved partial closure.

## Reporting and validation

Report cell-specific descriptive distributions and requested denominators with
failure bounds; exact recovery intervals are descriptive, without a new
hypothesis test or general power guarantee. Datasets are the sampling units;
methods on the same observations are paired measurements. Ten archived datasets
per cell do not constitute new independent replication.

Detailed observations, coordinates, parameters and provenance remain local.
Public exports contain small cell summaries and figures. Old scientific kernels,
defaults, serialization, p-values and empirical evidence gates remain unchanged.
The installed fixture uses invented raw data and untrained random NumPy encoders,
with Torch/SNPio imports blocked. It is engineering evidence only. Final closure
requires Python 3.11/3.12 tests, packages, exact-head CI and preservation checks.
No author contact is performed.
