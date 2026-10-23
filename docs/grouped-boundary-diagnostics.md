# Grouped boundary-rejection diagnostic

This checkpoint diagnoses the experimental likelihood-gain test without changing
production estimation, simulation, clustering, defaults or historical p-values.
Known generating parameters are available only for simulated data. Their reference
arm is a diagnostic, not an empirical calibration replacement, a selected
correction, species significance or a general size guarantee.

## Prespecified comparison

Source: c91b4fa1fd1c8543ed9a4dca58fbe33d4805f9d9 (0.31.0.dev0).
Diagnostic runtime: 0.32.0.dev0. All 132 existing modules other than version
metadata and CLI dispatch remain byte-identical. Diagnostic modules are additive.

Audit the five archived Gaussian scenarios: independent, correlated, zero
conditional covariance, rank-two conditional covariance and weak full-rank
conditional covariance. The 125 observed fits and 12,375 retained null fits
are historical evidence and are not rerun or overwritten.

Freeze sources, software versions, seeds and settings before fresh generation:

- 25 fresh observations/scenario, master seed 20261116; smoke seed 20261117.
- 64 independent groups, fixed non-informative counts of 1–4 morphology
  specimens/group, one ecology row/group, four continuous traits/view.
- Morphology standardization/PCA on equally weighted raw group means, two PCs/view,
  four joint PCs and equal view weights.
- Full-covariance GMM, K=1..4, minimum hard size 5, ten starts, model seed 71323.
- Alpha=0.05, 99 fitted-null draws and 99 known-law draws/observation.
- One shared observed fit: 125*(1+99+99)=24,875 new pipeline fits.
- No replacement observations/draws; at most four one-thread scenario jobs.

Both arms preserve the layout, counts and identities. Paired draws use the same
Gaussian innovations through each law's conditional/residual sampling factors.
These are common random numbers, not identical raw draws. Each arm's draw seeds
are independent of observed-data seeds and other draw indices. Coupled arms do
not supply independent evidence twice.

The fitted arm invokes unchanged empirical calibration. The known-law arm uses
analytic mean, joint between-group covariance and within-morphology covariance.
It copies observed archive bytes rather than fitting the observation again.
Every draw regenerates raw data and refits ingestion, group-mean standardization,
view PCA, aggregation, fusion, joint PCA and the complete GMM search.

## Questions and diagnostics

Given ecology, a morphology group mean has conditional covariance C + R/n:
C is conditional between-group covariance, R is within-morphology covariance,
and n is specimen count. At C=0, unequal counts still produce unequal precision.
A fitted positive C can change the relative variance across counts; that is a
hypothesis to examine, not an assumed cause.

Retain true/fitted conditional spectra, ranks, errors in true-within-covariance
units, within-covariance error and predicted conditional variance contrasts for
counts 1 versus 4. Compare these with tails/clustering. Independently audit dense
raw likelihood and constrained optimality. Evaluate the same raw observations
at the known generating parameters too; a lower negative log likelihood than
the accepted fit flags a feasible-parameter optimization concern. This is not
a global-optimization certificate or a solver repair.

For every pipeline inspect candidate likelihoods/eligibility, AIC/BIC, maximum
eligible likelihood gain, iterations, covariance eigenvalues relative to the
unchanged regularizer, retained PCA variance and descriptive association between
membership and specimen counts. Cramer's V has no post-selection p-value here
and is not a causal or biological partition test. The best-restart size-screen
behavior and earlier numerical issue remain unchanged.

The global statistic remains T = 2*(max_eligible log L_K - log L_1).
AIC component selection is separate. Retain BIC as a descriptive comparator from
the same candidate fits; do not choose a test by its p-value.

## Inference and uncertainty

Use upper tails including ties and (1+exceedances)/(1+99), with alpha=0.05.
Conditional binomial intervals describe simulation-tail uncertainty, excluding
null-estimation uncertainty. Rejection-rate Wilson intervals are pointwise and
unadjusted across scenarios. Paired discordances/p-value differences are
descriptive. Population-level uncertainty for null-pipeline characteristics is
calculated after averaging draws within each observation; generic pooled audit
descriptors do not supply across-observation standard errors. Twenty-five
observations/case limit conclusions, including those
about the known-law arm.

Failed point/null fits remain in accounting. Incomplete tests withhold
p-values/decisions and receive no replacements. Separate missing outcomes from
zero rejection and retain unavailable-outcome bounds. Count shared observed
fits once.

[Andrews (2000)](https://doi.org/10.1111/1468-0262.00114) motivates examining
boundary behavior; it is not a theorem establishing failure, validity or a
correction for this pipeline. Paired differences cannot alone identify a unique
cause or validate an empirical replacement. No threshold, solver budget,
variance floor, K search, covariance family, PC count or restart policy is tuned.

## Reproduction and audits

Install the clustering extra. Python 3.11/3.12 remain supported.

    super-deli diagnose-grouped-boundaries examples/grouped-boundary-diagnostics/study.json \
      --case zero_conditional_gaussian --output-dir /path/to/study/zero_conditional_gaussian
    # Run the other four scenarios into siblings.
    super-deli analyze-grouped-boundaries /path/to/study \
      --archived-study /path/to/aic-grouped-20261023/study --output-dir /path/to/new-audit
    python examples/grouped-boundary-diagnostics/smoke.py \
      --study examples/grouped-boundary-diagnostics/smoke.json --output-dir /path/to/new-smoke

The no-refit auditor checks reference covariances independently of the factory,
dense fitted-null likelihood/constrained convergence, regenerated raw arrays,
counts/identities, archive checksums, PCA moments/projections, fusion, every
candidate density/membership/criterion, seeds, point/null statistics, exact tails
and interval equations. Shared bytes and unique fit accounting are verified.
Solver checks do not certify a global likelihood optimum.

Portable results/figures accompany the report. Full arrays, candidates, draws,
failures, logs and the frozen runtime remain external. Preservation is not public
raw-data deposition or a container image.
