# Larger Gaussian false-split validation

This benchmark validates the unchanged matched separate-encoder/shared-GMM
calibration. It adds a batched study runner, not a new clustering model. The
production [calibration command](neural-calibration.md), legacy pilot schema,
defaults and historical p-values remain unchanged. Python 3.11/3.12 and the neural
extra are required.

Exact archival replay additionally requires the recorded pilot runtime: NumPy
2.2.6, SciPy 1.15.3, scikit-learn 1.6.1, Torch 2.15.0 and threadpoolctl 3.6.0.
The binder rejects a different runtime or changed scientific source; the package's
general dependency ranges do not promise bitwise archival parity across versions.

## Fixed study

Generate 400 fresh independent datasets at N=64: 200 each under the existing
independent-view and correlated-view Gaussian laws, with four continuous
morphology and eight individual ecology traits and feature-noise SD 0.15. Use
complete measurements from the same independent individuals. The Gaussian laws
are known simulation controls, not empirical species labels.

Each dataset receives 99 raw fitted-null draws. Both the native neural model and
the benchmark-only per-view PCA/shared-GMM reference use identical raw inputs,
fitted null parameters and draw streams. Fit the joint raw mean/full ML covariance
before learning, hold it fixed within a test, and refit ingestion, scaling,
both unchanged encoders (or view PCA), fusion and the complete GMM search in every
draw. No observed network weights are reused for null fits.

Keep neural seed 0, mixture seed 71323, two coordinates per view, shared
covariance, BIC K=1..4, minimum hard size five, ten starts, max_iter 500, tol 0.001
and reg_covar 1e-6. Each encoder retains 500 optimizer updates. The raw null has
the existing rank/singularity checks, without a new covariance floor or ridge.
Use the unchanged BIC-improvement statistic, plus-one tails including ties, and
alpha=0.05. Failed draws are retained without replacement; incomplete inference
is withheld. No seed, architecture, covariance-family or threshold search occurs.

RNG paths start with `[20261108, 690]`, followed by phase, law, replicate and
purpose. Main, preflight and diagnostic paths are disjoint; the two arms share
their assigned raw draws. Archived exact replays retain their original seeds.
The source/runtime/configuration declaration is frozen before generation.

The 400 observations run in eight fixed batches of fifty, with 25 datasets from
each law per batch. Final rates use all planned observations. No stopping or
configuration change is based on interim rates.

## Archived diagnostic and budgets

Keep the archived `gaussian_independent_0009` p=0.05 result unchanged. A separate
999-draw test on that fixed dataset uses a fresh frozen simulation stream and
both representations. It is a selected-case Monte Carlo precision diagnostic,
excluded from fresh-data rates. It does not establish that the original decision
was wrong or that the procedure is generally calibrated.

Replay that complete original 99-draw calibration and
`gaussian_correlated_0000` with their original seeds, checking scientific arrays,
candidate results, statistics and tails. Runtime and package-version metadata
are not equality targets.

The main study plans 82,400 candidate searches and branch trainings, with
41,200,000 optimizer updates: 80,000 fresh-data searches, 400 archived replay
searches and 2,000 sensitivity searches. Independent preflight uses two fresh
99-draw observations per law and one additional independent-view 999-draw
observation: 2,800 searches/trainings and 1,400,000 updates. Installation and test
fixtures are invented and excluded from scientific budgets and rates.

## Running the frozen workflow

The archive root is the completed initial calibration pilot, containing
`completion.json`, `frozen.json`, its original inventories, and the bound cases.
An output directory must be new and contain the approved plan. Detailed source
and per-fit files stay local.

```bash
python -m pip install '.[neural]'
mkdir /path/to/new-study
cp /path/to/approved-plan.md /path/to/new-study/approved-plan.md

super-deli benchmark-neural-false-splits bind \
  --archive-root /path/to/completed-pilot --output-dir /path/to/new-study
super-deli benchmark-neural-false-splits freeze \
  --archive-root /path/to/completed-pilot --output-dir /path/to/new-study
```

Run and audit `--phase preflight --group preflight`, then `finalize --phase
preflight`. The `gate` action requires `--staging-bytes` measured from actual
retained engineering evidence and generated workspace files; do not pass zero
for a real study with retained staging. The gate independently measures complete
law/N/draw-count strata, charges two staging copies and adds 25% allowance plus
a 128 MiB reserve. Do not begin main work after a failed gate.

After the gate passes, run then audit each main group `batch_00` through
`batch_07`, in order, followed by `extras`. The `run` action accepts `--workers 1`
or `--workers 2`; retain single-thread numerical libraries. For example:

```bash
super-deli benchmark-neural-false-splits run \
  --archive-root /path/to/completed-pilot --output-dir /path/to/new-study \
  --phase main --group batch_00 --workers 2
super-deli benchmark-neural-false-splits audit \
  --archive-root /path/to/completed-pilot --output-dir /path/to/new-study \
  --phase main --group batch_00
```

Finally use `finalize --phase main`, then `report`. Existing immutable results
are verified, not overwritten. A completed work unit is skipped only after its
source-bound identity and full inventory match. An interrupted unit is preserved
and reported unavailable rather than silently retried; its unknown work counts
remain explicit. A `pause` request is honored between current work units, and
`resume` archives the request before continuing. A numerical/source/resource
`STOP.json` requires an explicit amendment, not `resume`.

## Interpretation and verification

Eight primary rates distinguish ordinary BIC selection of K>1 from global
Gaussian rejection, separately for both laws and representations. Report exact
marginal 95% intervals, simultaneous intervals across eight rates, failure
bounds, and Monte Carlo standard errors. Four paired neural-minus-PCA contrasts
retain missing-pair accounting. A rate's Monte Carlo standard error concerns
finite study repetitions; a test's conditional tail interval concerns its finite
null draws. They measure different uncertainties. A zero observed rate does not
imply zero uncertainty, even if its plug-in standard error is zero.

Numerical audits check each actual optimizer update, raw estimates/draws,
identities, complete frozen configurations, transformations, all retained
candidate densities/memberships/BIC/eligibility, replay parity and exact tails.
Storage and runtime ceilings apply cumulatively across batches: at most two
one-thread CPU workers, 24 GiB new retained evidence, 8 GiB free, 40 serial hours
for main execution plus independent audit, and two serial preflight hours.
Projection receives no parallel-speed credit. Tool dependencies/caches are
separate from evidence quota but consume the free-space reserve.

An interval containing 0.05 is not proof of calibration. Results apply to the two
tested Gaussian laws, N=64 and the fixed settings; p-values remain approximate
fitted-model departure tests. No selected-K, cluster-specific, species-boundary
or general size guarantee follows. This checkpoint adds no stability analysis,
empirical fitting, new alternatives, known-parameter arm or author contact.
