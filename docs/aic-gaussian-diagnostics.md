# Gaussian-control AIC component-selection diagnostic

This checkpoint investigates why AIC sometimes selects several Gaussian mixture
components when the simulated data have no discrete generating groups. It adds
an isolated benchmarking harness, independent evidence audits and reporting.
Production fitting, simulation generators, defaults, historical p-values and
the existing best-likelihood-start-then-minimum-size policy are unchanged.

## Frozen design

`examples/aic-gaussian-diagnostics/study.json` declares 300 fresh raw observations:
50 each under independent, correlated and zero-conditional hierarchical Gaussian
laws at 64 and 256 localities. Each locality has 1–4 morphology specimens, four
continuous morphology traits and one row of four ecological traits. Counts are
independent and non-informative. No species labels or discrete groups are used.
The unchanged generators and analytic generating parameters are shared with the
preceding grouped calibration studies.

Three paired arms share counts, latent locality effects, ecology and Gaussian
innovations:

1. Existing group-mean PCA pipeline with ten mixture starts.
2. A deliberately altered simulation law: let `r = mean(1/n_i)`, where `n_i`
   is the training specimen count. Multiply each within-locality morphology
   residual by `sqrt(n_i*r)`. The covariance of the residual group mean is
   `r*W` at every locality, equal to the average baseline `W/n_i` covariance.
   Counts and identities remain unchanged. This is a diagnostic control, with
   no proposed empirical variance correction.
3. The original data/pipeline with 100 mixture starts.

All arms retain two PCs/view, four joint PCs, equal weights, a full covariance
per component, `K=1..4`, minimum distinct hard size five, model seed 71323,
regularization `1e-6`, tolerance `0.001` and 500 iterations. AIC and descriptive
BIC choices come from the same complete candidate pool. The 100-start arm
retains existing size-screening behavior, so changes can include eligibility
changes as well as improved optimization.

Another 300 independent controls (150 per training sample size) draw four scores
directly from `N(0,I/4)` and use the identical ten-start mixture search. They
remove ingestion/PCA but also change the covariance structure. They are separate
finite-sample controls, not paired observations or a uniquely causal PCA ablation.

Fifty original power-study Gaussian control point fits are replayed and compared
against their saved candidate arrays and transformations. The total main budget
is 900 raw pipelines + 300 score searches + 50 replays = **1,250 searches**.
The separate preflight uses 25 searches. Development and installed smoke tests
are additional verification work, recorded separately. No significance or
stability resampling occurs in this checkpoint.

## Independent prediction

Every fresh training observation has 2,048 independent held-out units, with
independent counts and innovations. Raw arms share these held-out innovations;
the equalized-noise arm uses the **training** `r`. Held-out identities are distinct.
Each unit passes through the saved training standardization, view PCs, view
normalization, fusion, joint PCs and candidate mixture parameters. Nothing is
fit on held-out data, and prediction never changes component selection.

Retained evidence includes held-out raw arrays, frozen-map scores, each candidate's
individual log densities, mean log-density gain over `K=1`, and conditional
Monte Carlo uncertainty. Comparisons are made within each fitted score space.
They describe prediction in that learned representation, not the probability
of the complete original biological measurements.

## Why examine both criteria and prediction?

For four-dimensional full-covariance mixtures, the parameter count is
`q_K = (K-1) + 4K + 10K = 15K-1`: 14, 29, 44 and 59 parameters.
AIC selects the eligible minimum of `-2*logL + 2*q_K`; BIC uses
`-2*logL + q_K*log(N)`. Each added component must gain 30 units of twice
log likelihood to offset its AIC penalty. At 64 training units, a four-component
fit has 59 parameters, so finite-sample flexibility deserves particular scrutiny.
Reconstructing these calculations does not establish a global likelihood optimum.

AIC is motivated by predictive approximation, not a test that Gaussian data
contain multiple biological populations. Finite Gaussian mixtures also have
nonregular/singular configurations. The independent prediction controls check
the observed behavior without assuming ordinary regular-model asymptotics.
See [Akaike (1974)](https://www.ism.ac.jp/akaikememorial/aic-e.html) and
[Watanabe (2010)](https://www.jmlr.org/papers/v11/watanabe10a.html).
This checkpoint implements no Bayesian criterion, AICc correction or new test.

## Evidence, uncertainty and limitations

The primary event is AIC selecting `K>1`. Eighteen raw-law/sample-size/arm rates
and two score-control rates form the 20 prespecified primary endpoints. They
receive exact pointwise 95% intervals, Bonferroni simultaneous 95% intervals,
Monte Carlo standard errors and explicit failure denominators/bounds. Twelve
paired baseline-minus-control comparisons use exact discordance bounds adjusted
over those comparisons. These intervals quantify variation across independent
simulated datasets; they are not species uncertainty.

The source audit covers 2,550 archived observed candidate pools, kept separate
from fresh observations. Existing significance results are copied read-only;
their reference draws and p-values are never recalculated. Candidate diagnostics
include density/membership reconstruction, likelihood gain versus penalties,
covariance eigenvalues relative to regularization, hard sizes, entropy, iteration
status, specimen-count associations and extremes under the one-component model.
No extreme observation or failed fit is discarded or replaced.

Independent numerical auditing reconstructs the Gaussian simulation law, residual
intervention, IDs/counts, saved PCA transformations, held-out projections,
candidate densities, memberships, criteria, selection and reporting totals.
The projection audit averages raw specimen values before applying the affine
map; the writer projects specimen rows before aggregation. A separate
standard-library checker verifies archive/member bytes, criteria and totals.
Neither audit refits models. Infrastructure failures retain partial evidence and
withhold incomplete results.

The variance intervention includes the pipeline's response to altered group means;
it does not isolate PCA adaptivity. Direct score controls change the score law
as well as removing preprocessing. More starts can change minimum-size eligibility.
Mechanisms can coexist. No configuration is promoted and no threshold, p-value,
null distribution or species-boundary claim changes.

## Reproduce locally

Install Python 3.11 or 3.12 with `.[clustering,dev]`. Regenerating the report
figure additionally requires Matplotlib. The primary study preserves
the preceding numerical stack; exact versions and source hashes are retained
locally alongside the frozen seeds. Installed workflow verification:

```bash
python examples/aic-gaussian-diagnostics/smoke.py --output-dir /absolute/new/smoke
```

The CLI provides `benchmark-aic-gaussian`, `analyze-aic-gaussian`,
`audit-aic-gaussian-sources`, `replay-aic-gaussian` and `preflight-aic-gaussian`.
Use `--help` for required paths. Audit the full-settings probes and one archived
replay, then obtain a matching measured storage preflight before the large run.
Full runs require that preflight; they never silently shrink a failed design.
At most four single-thread processes are allowed.

All detailed evidence remains in verified lossless local archives. A measured
projection uses 25% growth plus 64 MiB reserve, a 1 GiB checkpoint artifact cap
and at least 8 GiB free space. Only code, documentation, frozen settings, figures
and small aggregates are published. See the [completed diagnostic report](aic-gaussian-diagnostics-results.md) for
outcomes and verification.
