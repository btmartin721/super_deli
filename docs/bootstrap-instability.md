# Diagnosing small-sample predictive-bootstrap instability

This checkpoint diagnoses the existing shared-covariance predictive-screened AIC
workflow. It introduces no production estimator, selection rule, resampling law,
default change or p-value. Detailed inputs, fitted parameters, draws and private
indexes remain in verified local archives.

## Frozen comparisons

First diagnose all 40 prior observations and 3,960 retained draws without refitting.
The 792 prior probe draws are outside this cohort. Reconstruct distinct-locality
feasibility, full/fold eligibility, positive pooled held-out gain and full-data AIC
ranking. Keep their overlaps and a first-failing-gate classification. Total size
feasibility, generating-partition feasibility and fitted-partition eligibility
are different questions.

Generate ten fresh independent datasets in each of four cells: four groups at
N=64 with signal 3 or 4.5; four groups at N=256 with signal 3; and independent
Gaussian N=64 controls. Use the unchanged hierarchical continuous-data simulator,
1–4 morphology specimens and one ecology row per locality, four traits/view,
group-mean morphology PCA, two PCs/view, four joint PCs, equal view weights,
shared covariance, K=1..4, minimum five distinct original IDs, ten starts and
model seed 71323. All generating labels are held outside ordinary fitting and
selection.

For each observation, compare the baseline point plus 25 whole-locality draws,
a unique-support control on those same draws, and 100-start point/first-five-draw
fits. Remove duplicate localities in first-occurrence order for the unique
control, retaining every original member and the original fixed fold. This
changes preprocessing, likelihood sample size and optimization ordering as well
as frequency weights; it does not identify a single causal mechanism or define
a replacement bootstrap. Compare all three arms on the paired first-five prefix.

Fresh budget: 13,680 complete K searches. Replay ten prior N=64 observations'
points and first two draws exactly: 180 searches. Run 1,080 additional forced-K=4
truth-initialized optimization diagnostics on baseline scores, after ordinary
fits. Original-ID sizes determine their eligibility; missing groups/prerequisites
remain unavailable. These diagnostics never enter selection or stability.

## Evidence and checks

Retain source/raw identities, fold IDs, occurrence counts, transformations,
candidate parameters, validation densities, predictions and diagnostics. Trace
generating-axis retention through raw means, standardization, view PCA, profiles,
fusion and joint PCA; evaluate all-original and OOB projections separately.
Compare likelihood gains with AIC penalties only within an identical fitted score
space and sample measure. Report changes after label-invariant matching so stable
K cannot be mistaken for stable assignments.

Read-only gate-removal and equal-distinct-validation-weight counterfactuals keep
models fixed and preserve original decisions. If candidate parameters were never
fitted, a counterfactual requiring them is unavailable rather than invented.

Independent audits reconstruct raw laws, identities, PCA, projections, shared
mixture densities, criteria, distinct sizes, initialization and gates without
refitting. A separate standard-library verifier checks archive/member bytes,
fold leakage, selection and counters. The separately seeded full-settings probe
contains 1,368 searches and 108 forced fits. A measured storage gate uses archive
plus manifest sizes, explicit replay allowance, 25% growth, 128 MiB reserve,
a 3 GiB new-evidence cap, 8 GiB free and at most four single-thread workers.
Runtime guards retain partial evidence after failures; no replacement draws.

## Commands

```bash
super-deli archive-bootstrap-instability PRIOR_STUDY --output-dir archival-audit \
  --scratch .scratch --workers 4
super-deli benchmark-bootstrap-instability examples/bootstrap-instability/preflight.json \
  --output-dir probe --workers 4
super-deli analyze-bootstrap-instability probe --output-dir probe-audit \
  --scratch .scratch --workers 4
python examples/bootstrap-instability/check_diagnostic.py probe --output probe-check.json
super-deli preflight-bootstrap-instability examples/bootstrap-instability/study.json \
  --probe-study probe --probe-audit probe-audit --probe-check probe-check.json \
  --artifact-root LOCAL_ROOT --output-file storage-gate.json
super-deli benchmark-bootstrap-instability examples/bootstrap-instability/study.json \
  --preflight storage-gate.json --output-dir LOCAL_ROOT/study --workers 4
super-deli analyze-bootstrap-instability LOCAL_ROOT/study --output-dir study-audit \
  --scratch .scratch --workers 4
python examples/bootstrap-instability/check_diagnostic.py LOCAL_ROOT/study --output study-check.json
```

All output directories must be fresh. Use absolute paths outside the repository
for detailed study evidence. The installed smoke is
`examples/bootstrap-instability/smoke.py`; its smaller optimizer/draw budgets
verify engineering only. Python support remains 3.11/3.12.

## Interpretation

Original datasets are the independent simulation units. Draws and folds are
repeated measurements; five/five splits are not additional biological replicates.
Per-dataset exact binomial intervals describe only conditional finite-draw
precision for individual events. They are unadjusted, not simultaneous,
biological confidence or species probabilities. Show failure bounds and all
requested/available denominators. Ten-per-cell paired effects and ranges are
descriptive. There is no new population-performance, causal, species-boundary,
significance or generally valid correction claim. No observed setting is promoted
automatically. See the [results](bootstrap-instability-results.md).
