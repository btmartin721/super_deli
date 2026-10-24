# Bounded grouped AIC/likelihood power and recovery validation

This benchmark evaluates the existing experimental full-covariance grouped
AIC/likelihood workflow. It preserves production fitting/sampling code, BIC and
other defaults, previous p-values, inputs, checkouts and studies. Constructed
groups supply evaluation truth; they are not empirical species identities.

## Frozen design and fit budget

The declared design contains 400 new independent observations:

| Generating law | Localities | Raw signal multiplier | Observations |
| --- | --- | --- | ---: |
| Complementary four groups | 64 and 256 | 1.5, 3, 4.5 at each N | 50 per cell; 300 total |
| Independent hierarchical Gaussian | 64 and 256 | None | 25 per cell; 50 total |
| Heavy-tailed and continuous-cline stress | 64 | Existing stress laws | 25 per law; 50 total |

Signal 3 exactly reproduces the existing complementary generator. The benchmark
wrapper changes only alternative mean shifts for 1.5 and 4.5. Noise, balanced
labels and random non-informative morphology counts remain unchanged. Each
observation has 1–4 morphology specimens/locality, one ecology row/locality and
four continuous traits/view. All locality/specimen links are explicit.

Fix group-mean morphology PCA, two PCs/view, four joint PCs, equal view weights,
full GMM covariance, K=1..4, minimum distinct cluster size 5, ten starts and model
seed 71323. Keep existing regularization, solver budgets and start eligibility.
Freeze master seed 20261120, all observation/reference/draw streams and source
fingerprints before generation; numeric seeds are identifiers, not run dates.
Separate development/preflight seeds supply no replacement study observations.

Fit the unchanged schema-2 conditional-PSD hierarchical Gaussian raw null once
per observation. Hold its parameters/counts fixed for 99 independent parametric
draws; regenerate raw inputs and refit ingestion, group means, standardization,
view PCA, aggregation, fusion, joint PCA and the complete GMM search in each.
The primary statistic is maximum eligible twice-log-likelihood gain over K=1.
Use the plus-one upper tail including ties and alpha=0.05; no chi-squared reference.
Share candidate fits and draws for AIC/BIC component selection and the descriptive
BIC departure statistic. Do not choose a statistic by its resulting p-value.

Calibration requires 400 * (1+99) = **40,000 pipeline fits**. On only the first
five observations in each of six alternative cells, run a separate verified
point replay and 25 whole-locality stability draws: **30 replays + 750 refits**.
The total is **40,780 planned pipelines**, excluding tiny development/preflight
runs. Use at most four single-thread jobs. Labels never enter fitting, tuning,
model selection or resampling. No truth-initialized optimization is included.

Retain failed observations/draws and partial evidence without replacements;
withhold incomplete calibrated p-values. Stability remains separate from
significance and retains complete group members and draw multiplicities.

## Outcomes and uncertainty

The three primary binary endpoints in each alternative cell are global likelihood
rejection, AIC choosing K=4 and their joint event: **18 primary rates**. Report
counts, availability, requested denominators, failure bounds, exact pointwise
95% binomial intervals, Monte Carlo standard errors and Bonferroni exact intervals
with 95% simultaneous coverage over these 18 rates. Fifty independent datasets
per cell have worst-case MCSE about 7.1 percentage points; intervals can be wide.
The joint event does not make K or memberships statistically significant.

Report selected-K distributions and unconditional adjusted Rand agreement with
generating labels: mean, median, quartiles and across-observation MCSE. Summaries
restricted to rejected datasets are secondary and explicitly conditional.
Paired BIC comparisons are descriptive with uncertainty, not a new selection
rule. Do not compare absolute criteria between different score spaces.

Gaussian controls screen rejection and multiple-component selection separately;
25 observations/cell do not replace the completed 500-per-scenario validation.
Heavy tails and clines have no discrete truth. Their rejection is Gaussian-model
departure, not a species/type-I error. Multiple components can approximate a
continuous distribution and do not establish multiple species.

For the fixed stability subset, report availability, selected-K frequencies,
assignment agreement and co-assignment error against generating truth. Summarize
these per observation, then across observations; locality pairs are not independent
study replicates. With 25 draws, frequencies are coarse and OOB pair denominators
can be small or zero. Failure bounds are not confidence intervals. Stability
measures conditional repeatability, not species confidence.

## Evidence, storage gate and independent checks

Keep large evidence local in lossless per-observation archives. Verify every
member name, size and original file byte before removing only the new study's
redundant working copy. Reject unsafe paths, duplicate/nonregular members and
checksum changes. Preserve failed compaction evidence. Prior studies are untouched.
Numerical audits materialize one observation per worker in checked scratch space;
archived absolute provenance remains unchanged and is safely rebased for reading.

Before the full run, execute independent full-settings probes at both sample
sizes, including stability, and independently audit them. Measure calibration
and stability retention separately, account for the frozen non-informative counts,
add 10% growth and 256 MiB reporting/audit reserve, and include existing checkpoint
evidence. Require the projected complete evidence to fit the **8 GiB artifact
cap** while leaving **8 GiB free**. Unexpected growth stops execution and preserves
partial evidence; it never silently reduces the dataset count or drops records.

Audit generating arrays, truth/identities/counts, raw-null dense likelihood and
constrained convergence, view-PCA moments/projections, fusion, every candidate
density/membership/criterion and exact upper-tail accounting. Independently verify
stability point replay, resampling seeds/multiplicities, source members, PCA,
candidate selection, partitions and co-assignment/OOB counts. The unchanged
production stability export does not contain per-resample Gaussian parameter
arrays; complete density audits cover calibration fits and replay points, not
those unexported stability parameters. Numerical checks do not certify global
likelihood optimization. The earlier numerical issue/restart behavior is unchanged.

A separate Python-standard-library checker reconstructs candidate criteria,
component selection and both statistics/tails from archived records. No auditing
step refits a model. Publish only code, documentation, figures and small aggregate
records/settings; detailed tables, private artifact indexes, arrays, models and
logs remain local. Fit/runtime source fingerprints are frozen and retained.

## Reproduction

Use Python 3.11/3.12 with `super-deli[clustering]`. Freeze the recorded numerical
versions for reproducing the experiment. Figures additionally require Matplotlib
3.7 or newer. Use a new output directory for each immutable run:

```bash
super-deli benchmark-grouped-power examples/grouped-power-recovery/preflight.json \
  --output-dir /path/to/artifacts/probe --workers 2
super-deli analyze-grouped-power /path/to/artifacts/probe \
  --output-dir /path/to/artifacts/probe-audit --scratch /path/to/artifacts/.scratch --workers 2
super-deli preflight-grouped-power examples/grouped-power-recovery/study.json \
  --probe-study /path/to/artifacts/probe --probe-audit /path/to/artifacts/probe-audit \
  --artifact-root /path/to/artifacts --scratch /path/to/artifacts/.scratch \
  --output-file /path/to/artifacts/storage-preflight.json
super-deli benchmark-grouped-power examples/grouped-power-recovery/study.json \
  --preflight /path/to/artifacts/storage-preflight.json \
  --output-dir /path/to/artifacts/study --workers 4
super-deli analyze-grouped-power /path/to/artifacts/study \
  --output-dir /path/to/artifacts/audit --scratch /path/to/artifacts/.scratch --workers 4
python examples/grouped-power-recovery/check_tails.py \
  --study-root /path/to/artifacts/study --audit-root /path/to/artifacts/audit \
  --output /path/to/artifacts/independent-tails.json
python scripts/report_grouped_power.py /path/to/artifacts/audit/summary.json \
  --output-dir /path/to/artifacts/report
```

The installed smoke uses one observation/cell, one null/stability draw and two EM
starts, with separate seeds. It verifies workflow behavior, not performance.
CI places fixtures under `RUNNER_TEMP`, avoiding the Ubuntu 26.04 `/tmp` tmpfs
that is smaller than the production benchmark's unchanged 8 GiB guard.

This is a fixed-design, approximate fitted-Gaussian departure validation. It
does not establish broad size control, equivalence, selected-K significance,
empirical biological accuracy or species boundaries. No default is promoted.
