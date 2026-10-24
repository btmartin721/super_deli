# Larger grouped likelihood-gain false-positive validation

This checkpoint measures rejection under five declared hierarchical Gaussian
laws using more independent observed datasets. It preserves production estimation,
simulation, clustering defaults, ten-start screening, historical p-values and
all retained earlier evidence. It introduces no correction or species test.

## Frozen design

The study reuses the exact frozen 0.32.0.dev0 fit runtime from master
`9c336d137da70f5f2cd772f575daeaead10c3adb`; all 138 package modules are pinned.
Reporting is additive in 0.33.0.dev0. Generating laws are unchanged from the
[boundary diagnostic](grouped-boundary-diagnostics.md).

- 500 fresh independent observations for each of independent, correlated,
  zero-conditional, rank-two and weak-conditional Gaussian laws: 2,500 total.
- 64 independent localities, fixed non-informative 1–4 morphology specimens per
  locality, one ecology row/locality, four continuous features per view.
- Group-mean morphology standardization/PCA, two PCs/view, four joint PCs,
  equal weights, full component covariance, K=1..4, minimum hard size 5,
  ten starts, model seed 71323 and alpha=0.05.
- 99 fitted-null and 99 analytically known-law draws per observation. Paired
  arms use the same Gaussian innovations and counts, with one observed fit
  shared unchanged: 2,500*(1+99+99)=497,500 planned unique pipeline fits.
- Master seed 20261118; independent smoke seed 20261119. Data, reference-stream
  and draw seeds are frozen before generation and checked for collisions and
  overlap with retained earlier study seeds.
- At most four single-thread scenario jobs. No replacements, adaptive stopping,
  repeated fitting to repair outcomes, threshold tuning or stability bootstrap.

Every draw regenerates raw inputs and refits ingestion, raw group means,
standardization, view PCA, aggregation, fusion, joint PCA and the complete
prespecified GMM search. Null parameters are fitted once per observation and
held fixed within its fitted-null test. The known-law arm is only a simulation
control; its parameters cannot be supplied by unknown real biological truth.

## Primary outcomes and uncertainty

The fixed primary statistic is maximum eligible twice-log-likelihood gain
relative to K=1. Upper tails include ties, with p=(1+exceedances)/100. The complete
K search and unchanged eligibility screen operate in each reference draw.
AIC component selection is separate; BIC is retained descriptively and is not
selected as a test after looking at results.

There are ten primary rejection rates: five laws by two reference arms. Report
counts, availability, requested denominators, plug-in Monte Carlo standard errors,
pointwise exact Clopper–Pearson 95% intervals and Bonferroni exact intervals
with 95% simultaneous coverage across these ten rates. The adjustment concerns
simulation performance summaries; it does not alter any original p-value or
alpha. At a 5% rejection probability, 500 independent observations give an MCSE
of approximately 0.00975, or 0.975 percentage points. This is a planning value,
not a precision guarantee or an acceptable-error threshold.

For paired decisions report both/neither/one-arm-only counts and the known-minus-
fitted rate difference. Its conservative interval subtracts exact marginal
intervals for the two discordance probabilities, allocating half the error to
each interval. Independence between arms is not assumed. Also retain intervals
adjusted across five paired comparisons. Agreement is not an equivalence test.

Incomplete tests withhold p-values and decisions and receive no replacement.
Report conditional-on-availability intervals separately from all-requested
missing-outcome bounds. The latter are not confidence intervals. Conditional
tail intervals for each original test describe simulation uncertainty given its
observed statistic and fixed reference parameters; they do not cover estimated
null parameters or model misspecification.

## Audit and evidence

The streaming auditor reuses independent raw likelihood/constrained-convergence,
PCA/projection, generated-array, candidate-density/membership/criterion and exact-
tail checks. It verifies all retained archive members and identities, counts,
seeds and unique-fit accounting. It writes per-scenario compressed point tables
without retaining the whole study's point records in memory. Audit workers are
limited to four with one numerical thread each. The separate standard-library
checker recomputes both statistics and every tail independently of the numerical
package. Audits never refit a model.

Full inputs, parameters, draws, candidate models, failures and logs are retained
outside Git. Only small aggregate summaries, settings, figures and checksums
of published records accompany the published results; detailed audit tables remain local.
Existing archives and checkouts are fingerprinted before/after.
Source availability at checkpoint start is recorded explicitly; an earlier
source download already absent at that time is not claimed as preserved.

## Reproduction

Use Python 3.11 or 3.12 with the clustering extra. Figure generation additionally
requires Matplotlib 3.7 or newer. Freeze the recorded numerical
versions when reproducing the published fit experiment. A separate smoke uses
one dataset/law and one draw/arm, with the same 64-locality analysis settings.

```bash
super-deli diagnose-grouped-boundaries examples/grouped-false-positive/study.json \
  --case independent_gaussian --output-dir /path/to/new-study/independent_gaussian
# Run the other four declared cases into sibling directories.
super-deli analyze-grouped-false-positives /path/to/new-study \
  --workers 4 --output-dir /path/to/new-audit
python examples/grouped-false-positive/check_tails.py \
  --study-root /path/to/new-study --tails /path/to/new-audit/tails.csv \
  --output /path/to/independent-check.json
python scripts/report_grouped_false_positives.py /path/to/new-audit/summary.json \
  --output-dir /path/to/new-figures
```

This finite study estimates behavior only for the declared laws, sample design,
algorithm and 99-draw tests. It cannot establish uniform size control across
Gaussian parameters or misspecified biology, selected-K significance, species
confidence or anatomical accuracy. Numerical verification does not certify a
global likelihood optimum. The earlier numerical issue and restart-admissibility
behavior remain unchanged.
