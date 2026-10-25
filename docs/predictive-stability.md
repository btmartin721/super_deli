# Whole-locality predictive stability

This opt-in workflow asks whether grouped morphology/ecology clustering repeats
when the observed independent localities are resampled. It reruns the complete
shared-covariance, predictive-screened AIC analysis for each draw. The original
point analysis and its assignments remain separate. Existing full-covariance/BIC
defaults, significance procedures and historical p-values are unchanged.

## Scope and invocation

Require complete continuous specimen morphology linked explicitly to localities,
one complete continuous ecology row per locality, group-mean morphology PCA,
shared (`tied`) covariance, AIC, five whole-locality folds, one numerical thread
and the independence declaration. The point model retains `n_bootstrap: 0`;
resampling settings are separate and explicit. Population targets, missingness,
other modalities/families and ordinary unscreened shared-AIC stability remain
unsupported. This does not make shared covariance a default or choose a family.

From the repository, for a compatible dataset declaration:

```bash
super-deli cluster-integrative-predictive-stability dataset.json \
  --model-config examples/predictive-stability/model.json \
  --screen-config examples/predictive-stability/screen.json \
  --bootstrap-config examples/predictive-stability/bootstrap.json \
  --output-dir predictive-stability-run
```

The API is `super_deli.workflows.predictive_bootstrap.run_predictive_bootstrap`.
`PredictiveBootstrapConfig` requires `schema_version: 1`, `n_bootstrap` and an
unsigned 64-bit `random_seed`. Existing predictive CLI commands and serialization
contracts are unchanged. Output directories must be new; failures retain evidence.

## What is resampled

For N localities, draw N original locality indices with replacement. Keep every
morphology member/count and the single ecology record together. Fit one raw group
mean per occurrence, with multiplicities in standardization, both view PCAs,
profile normalization, fusion, joint PCA and mixture likelihood. The original-ID
minimum cluster size counts distinct localities, never copied occurrences.

Freeze the point analysis's original five-fold partition and full/fold optimizer
seeds across draws. All copies of an original locality stay in the same fold.
Refit every transformation and K candidate inside training folds. Validation
contributions retain draw multiplicity and are pooled over N occurrences; fold
means are not averaged equally. The unchanged screen requires full/fold candidate
eligibility and strictly positive pooled log-density gain over K=1, then minimizes
full-draw AIC among passing candidates and K=1, with smaller-K ties.

This estimates resampling stability conditional on that partition, seed rule,
observed within-locality sampling and model settings. It neither isolates a pure
biological variance component nor evaluates alternative partitions. Localities
must be independent; spatial dependence or related sampling can violate this.

Empty validation folds, inadequate independent training units, rank deficiency,
unresolved fits and numerical/projection failures make a draw unavailable. Keep
its original seed, partial evidence and attempted/unattempted search counts. Never
replace a draw, change the folds, reduce PCs or substitute K=1 after a failure.
Ordinary K-specific size exclusions retain the current candidate policy.

## Reading outputs

- `point/` contains the unchanged point workflow and original fold partition.
- Each `draw_XXXX/` retains raw ingestion, identities, model seeds, transformation
  parameters, complete candidate pools, validation densities, screen decisions,
  selected model and projections/predictions for all original localities.
- Each `draw_XXXX.json` records indices, integer multiplicities, out-of-bag indices,
  status, selected K, original-unit predictions and search counts.
- `stability.npz` stores coassignment, exact original-point-cluster recovery and
  selected-K counts with requested/successful denominators, failure bounds,
  conditional frequencies and Monte Carlo precision. Selected-K arrays use
  index K-1 through the largest point/draw K; unobserved trailing K are omitted.
- `stability.json` and `summary.json` give counts, conditioning and cluster summaries.
  Within-cluster cohesion appears alongside cross-cluster merging: selecting K=1
  repeatedly cannot be mistaken for preservation of separate point clusters.

`all_*` pair frequencies project every original locality, including training units;
resubstitution can make them optimistic. `oob_*` counts use only draws where both
original localities were absent from training and screening. Its pair-specific
requested and successful counts differ. Require at least ten successful both-OOB
draws to display a point frequency; otherwise retain counts, coverage and failure
bounds and store the point frequency as NaN. Cluster OOB averages use covered
pairs only and report covered/total pair counts; coverage is not guaranteed.
OOB evaluation reuses the observed units and is not independent biological truth.

After any unresolved draw, unconditional frequencies are unavailable; lower/upper
unknown-outcome bounds retain the requested denominator, and explicitly conditional
summaries describe successful draws only. Failure can bias those conditional
summaries. Monte Carlo standard errors and unadjusted exact binomial 95% intervals
quantify finite-draw precision of individual conditional simulation events; they
are not biological confidence intervals, simultaneous pairwise intervals, species
probabilities, p-values or confidence coverage for the true number of groups.

99 draws are a bounded validation budget, not a universal recommendation. Both-OOB
precision is particularly limited; use larger explicit budgets when appropriate.
The dense summaries also require a conservative pairwise-memory preflight.

## Reproducible bounded validation

See [the frozen design](../examples/predictive-stability/study.json) and
[preflight](../examples/predictive-stability/preflight.json). Forty independent raw
datasets span eight cells: five each under three N=64 Gaussian controls,
N=64 four groups at signals 3/4.5, N=256 four groups at signal 3,
N=64 unequal-variance four-group stress and continuous-cline stress. The unchanged
raw generators use 1–4 fixed non-informative morphology specimens/locality,
one ecology row, four traits/view, two PCs/view, four joint PCs and equal weights.
K=1..4, minimum distinct size five, ten starts, 500 iterations, tolerance 0.001,
regularization 1e-6 and model seed 71323 are prespecified. Generating labels enter
only evaluation after fitting, never selection.

Each dataset has one point workflow and 99 resampled workflows, six K searches
each: 24,000 searches. Fifteen exact archived six-search replays add 90 searches.
Eight separately seeded full-settings probes add 4,800 preflight searches.
No significance, truth-initialized fits, external test-set benchmark or tuning is
included. Five datasets/cell establish bounded correctness/reproducibility only;
bootstrap draws and folds are not independent simulated datasets. This does not
calibrate stability probabilities or precise population performance rates.

The numerical auditor independently reconstructs raw laws, weighted moments/PCA,
original-ID folds and sizes, Gaussian likelihood/membership, AIC/BIC, pooled gates,
projections, finite-draw intervals and retained denominators without fitting models.
`examples/predictive-stability/check_stability.py` separately verifies archive/member
bytes, draw/fold identities, criteria/gates and stability counts using the standard
library. Both preserve failure evidence. A reduced installed smoke uses two starts
and two draws; it does not replace the full-settings probes or study.

Keep detailed evidence local. The measured gate includes archive manifests, 25%
growth, a replay allowance and 128 MiB reserve; require a 4 GiB new-artifact cap,
8 GiB free, runtime guards and at most four single-thread workers. Publish only
source, fixed configurations, documentation, small aggregates and figures.
