# Diagnosing predictive-screen recovery at N=64

This checkpoint diagnoses the optional predictive-screened AIC workflow. It leaves
production fitting, sampling, selection, defaults, serialization and historical
p-values unchanged. The [preceding validation](predictive-aic-results.md) found
that the screen suppressed Gaussian-control partitioning but also all K=4 recovery
at N=64. Prediction quality and recovery of generating groups are distinct targets.

## Questions and archived evidence

The diagnosis distinguishes full-data eligibility, each training fold's size
constraint, validation log-density gain and final full-data AIC ranking. Every K
is inspected, rather than only the ordinary AIC winner. All 625 archived datasets
and 3,750 full/fold searches are checked without refitting. Each fold's independent
external sample is projected through its frozen training map and scored by every
fitted candidate. External data never influence selection or settings.

Two gate-removal counterfactuals are diagnostic calculations on the same retained
candidate pools. Eligibility-only omits predictive gain; gain-only retains full-data
eligibility but omits training-fold hard-size gates. Gain-only can therefore use
already fitted small-component fold models. Neither is offered as a production
selector, calibrated test or proposed new default. Numerical failures withhold a
result; they never become a valid K=1 selection.

Private evidence traces generating axes through raw group means, view PCA,
aggregation, fusion and joint PCA; component sizes, covariance spectra, weights,
conditioning and validation-unit contributions; training/validation group balance;
and raw-coordinate PCA subspace changes. Generating labels enter only post-fit
evaluation or the separately marked oracle diagnostic. Associations among these
quantities do not identify a unique causal mechanism.

## Frozen fresh design

Generate 100 independent raw datasets, 25 each:

| Law | Localities | Signal |
| --- | ---: | ---: |
| Complementary four groups | 64 | 3 |
| Complementary four groups | 64 | 4.5 |
| Complementary four groups | 256 | 3 |
| Independent hierarchical Gaussian control | 64 | none |

The archived study retains the weaker signal, additional controls and stress laws;
those omitted cells receive no fresh-validation claim here. Generating laws are
unchanged, using 1–4 specimens/locality, four continuous morphology traits/specimen,
four ecology traits observed once/locality, and fixed noninformative counts.

The model fixes group-mean morphology PCA, two PCs/view, four joint PCs, equal view
weights, full component covariance, K=1..4, minimum distinct hard size five, ten
starts, max_iter=500, tol=0.001, reg_covar=1e-6 and model seed 71323. Raw datasets,
independent external samples and five partition seeds are frozen before generation.
Each partition shuffles whole localities into five folds without truth stratification.
Every fold re-ingests raw inputs and refits the complete pipeline using training
units only. The existing deterministic per-fold model-seed rule is retained.

For each observation, share one full-data fit across five partitions and fit five
training folds per partition: **26 searches**, or 2,600. Repeat the full-data and
first partition's five fold searches with 100 EM starts, changing no other setting:
**600 additional searches**. The extra starts are a diagnostic arm, retaining the
existing best-start admissibility behavior. Reused full fits are copied as evidence,
verified byte-for-byte, and counted once in unique-fit accounting. Per-partition
logical six-search summaries are explicitly distinguished from unique study counts.

Add 25 exact archived six-search replays: five each from N=64 signals 1.5/3/4.5,
N=256 signal 3 and N=64 independent Gaussian. The main unsupervised budget is
**3,350 searches**; a search includes the K=1..4 candidate procedure, not one EM start.

## Separate truth-initialized diagnosis

For the 75 fresh alternatives, make six separate K=4 fits in their full and first
partition's fold score spaces: **450 single-K fits**. Initialize component means,
weights and full covariances using generating labels of training units only, then
run one explicit EM start with the retained settings. Preserve unsupported or
failed initialization and numerical outcomes without substitution.

These oracle parameters never enter ordinary AIC, screened selection or p-values.
Their likelihood, eligibility, agreement and validation/external prediction can
help diagnose local optima. Even improved recovery does not establish a deployable
solution. Main accounting is **3,350 searches + 450 single-K fits = 3,800 attempts**.

## Evaluation and uncertainty

Each fresh observation receives 2,048 independently generated external localities,
with disjoint identities and counts. Their outcomes never set thresholds, choose
folds or select models. Compare candidates' predictive gains within the same
training map and relative to its K=1 fit. Do not compare absolute likelihood,
AIC or BIC across separately learned maps.

The primary family uses the first partition: ordinary and screened choices under
ten and 100 starts in four cells, **16 rates**. Alternative events are K=4 recovery;
the control event is K>1 selection. Report exact pointwise and Bonferroni
simultaneous 95% intervals, dataset-level MCSE, availability and withheld bounds.
Eight prespecified paired comparisons cover ten-start screen versus ordinary AIC,
and screened 100 versus ten starts, in each cell, with adjusted exact discordance
bounds. The arms are paired, not independent samples.

Repeat-partition results are within-observation sensitivity. Report per-dataset
selection/passage counts and agreement; five partitions are not five independent
datasets or biological stability replicates. Give no fold-independence SE or CV
p-value. Gate counterfactuals, oracle outcomes, ARI, spectra, prediction gains,
signal retention and BIC remain descriptive, with means/medians/IQR and denominators.
Keep prediction, component count, generating-group recovery and species inference
separate. No significance or stability study is added here.

## Commands and local evidence

Use Python 3.11/3.12 and the `clustering` extra. Configurations are in
`examples/predictive-recovery/`. Relevant commands are:

```text
benchmark-predictive-recovery CONFIG --output-dir NEW_STUDY --workers 4 --preflight GATE
analyze-predictive-recovery STUDY --output-dir NEW_AUDIT --scratch SCRATCH --workers 4
diagnose-predictive-archives CONFIG --source-study PRIOR --output-dir NEW_DIAGNOSIS \
  --artifact-root BUDGET_ROOT --scratch SCRATCH --workers 4
preflight-predictive-recovery CONFIG --probe-study PROBE --probe-audit PROBE_AUDIT \
  --artifact-root BUDGET_ROOT --output-file NEW_GATE
replay-predictive-recovery CONFIG --source-study PRIOR --output-dir NEW_REPLAYS \
  --scratch SCRATCH --workers 4
```

The installed smoke exercises repeated partitions, shared fits, the 100-start
arm with a smaller test start budget, oracle separation and two audit paths:

```bash
python examples/predictive-recovery/smoke.py --output-dir /absolute/new-output
```

The full-settings preflight uses one observation/cell: **128 searches and 18
single-K fits**, separate from the main budget. Require its independent audit,
a measured 3 GiB new-evidence cap and at least 8 GiB free. Project verified lossless
retention with 25% growth plus 128 MiB reserve; enforce runtime space checks and
at most four single-thread jobs. Failed gates require a storage/design revision.
The original 2 GiB gate failed at 2.75 GiB projected retention. A separately
approved budget-only amendment raised the cap to 3 GiB. The replacement gate
reused the identical 146 audited probe fits without regenerating data or changing
seeds, settings or statistical rules; the original configurations and failed gate
remain retained locally.
Retain failed and unattempted fits explicitly without replacement. Preserve all
previous studies, inputs, checkouts and numerical anomalies.

Independent numerical audits reconstruct raw simulations, identities/counts,
partitions, preprocessing, projection, densities/memberships/criteria, constraints,
gate arithmetic, selections, exports, oracle initialization and fit counts. The
separate standard-library verifier checks archive/member bytes, decoded validation
vectors, counterfactuals, shared-fit identity, budgets and primary totals. Neither
checker refits models. Complete evidence stays local; publish only code, frozen
settings, documentation, figures and small aggregates. Figure reproduction needs
Matplotlib. Development, preflight and installed-smoke fits are reported separately.

## External-density audit arithmetic

The first fresh-data audit stopped on one external log-density difference of
1.29e-8 against its fixed tolerance. A separate 70-digit decimal calculation
showed that the retained model density differed by only 1.22e-10, while the
auditor's eigenvalue-based calculation differed by 1.28e-8. This checkpoint's
external-density audit now uses independent Cholesky triangular solves with the
original tolerance. A regression fixes that synthetic case and rejects a
deliberately corrupted density. Production fitting/scoring, all observed fits,
selections and p-values are unchanged. The original runtime, failed audit and
high-precision evidence remain local, with a separately fingerprinted audit
runtime. No scientific fits were regenerated for this numerical audit repair.
