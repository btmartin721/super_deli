# Bounded calibrated neural power and group recovery

This opt-in benchmark compares unchanged separate-encoder/shared-GMM calibration
with the existing per-view PCA/shared-GMM reference. Inputs are invented complete
matched morphology and individual ecology tables. It produces no species test,
new model, default, empirical cohort or historical p-value replacement.

## Fixed study

The 100 independent main observations contain 60 balanced complementary four-group
datasets (ten each at center multipliers 0.5/1/1.5 crossed with N=64/256), twenty
independent Gaussian controls (ten per sample size), and N=64 stress datasets:
ten unequal-spread, five heavy-tail and five continuous-gradient observations.
Morphology has four traits and ecology eight; original loadings, unit residuals
and independent feature noise SD 0.15 are retained. Only group centers change
across the three signal strengths. Gradient stress contains no discrete truth
partition. Labels/gradient coordinates enter evaluation only, after fitting.

Both arms retain two coordinates per view, equal influence, the shared covariance
GMM, BIC K=1..4, hard minimum size five, ten starts, mixture seed 71323 and neural
seed 0. Existing 500-update CPU float64 branch training remains unchanged. Each
observation fits one joint raw-feature Gaussian MLE before learning, holds it
fixed for 99 draws, and refits ingestion, scaling, representation, fusion and the
complete GMM search every time. Shared draws are paired arm measurements, not
independent observations. The plus-one global BIC-improvement test retains ties,
alpha=0.05 and failed-draw withholding.

There are 20,400 planned main searches, including two complete archival B=99
replays, and 2,200 separate full-settings preflight searches. Source/runtime/plan,
configurations, identity orders and the namespace [20261109,700] are frozen before
generation. Main and preflight streams are independent. Exact old calibration
replays exclude timing/version metadata but retain arrays, seeds, candidates,
statistics and inference parity. Old inputs, source modules and p-values remain
unchanged.

## Interface and evidence

```bash
python -m super_deli benchmark-neural-power --help
```

The archive root is the common parent containing the completed
`neural-calibration-20261107` and `neural-false-splits-20261108` studies. Use a new
output directory containing the approved plan as `approved-plan.md`; keep detailed
evidence outside the repository. Supply `examples/neural-power/study.json` when
binding/freezing the real study. Run actions in this order: `bind`, `freeze`,
`run --phase preflight`, `audit --phase preflight`, `gate --staging-bytes BYTES`,
`run --phase main`, `audit --phase main`, `report`. Every action requires
`--archive-root` and `--output-dir`; `--workers` may be one or two.

The dependency-free configuration fixes the scientific grid. Resource fields may
only tighten its approved bounds. The measured gate matches law/strength/N/draw
strata, charges serial execution/audit time without parallel credit, counts staging
and verification copies, and adds 25% projection allowance plus a 128-MiB reserve.
Main execution/audit must project below ten serial hours, separate preflight below
two serial hours, new retained evidence below eight GiB and free storage above
eight GiB. Tool environments/caches are excluded from the evidence quota but still
consume free disk space. Preserve failed gates and seek an explicit amendment;
settings/sample counts are not automatically reduced.

Whole calibration work units are sealed with source-bound inventories. A continued
run verifies and skips completed units; interrupted units retain unknown counters
and unavailable inference without replacement. `pause` is honored between units;
`resume` archives the request. Same-host live/foreign locks and orphaned workers
are protected. Numerical, source, integrity and cumulative resource guards retain
a STOP record. No convergence rescue, new seed or outcome-based extension occurs.

## Reporting contract

The six balanced signal cells and two arms contribute 36 primary rates: global
rejection, recovery (K=4 and ARI>=0.8), and their joint event. There are eighteen
paired neural-minus-PCA contrasts. Report exact marginal and simultaneous
intervals (36 rates / 18 contrasts), requested denominators, unavailable bounds,
Monte Carlo standard errors and conditional tail uncertainty. A failed null draw
withholds the global p-value; an available observed fit may still supply recovery.
Joint outcomes require both recovery and inference to be available.

Selected K, correct K alone, underselection, search-boundary selection and ARI
are retained separately. The fixed K<=4 search cannot evaluate overpartitioning
beyond four. Gaussian splits and Gaussian rejections are distinct endpoints.
Heavy-tail/gradient rejection means departure from the Gaussian reference,
without a species or Gaussian-null false-positive interpretation. Unequal-spread
truth permits separate recovery evaluation. Secondary marginal intervals are
descriptive, outside the primary multiple-comparison family.

Ten observations per signal cell give broad intervals even with perfect recovery;
five stress observations support only descriptive checks. The results cannot
establish a precise operating threshold, neural superiority, general calibration
or species validity. Absolute BIC is compared only within one score space.

`examples/neural-power/smoke.py` executes the entire installed lifecycle on a
separately labelled invented N=32 fixture. Full inputs, weights, update checks,
null draws and specimen results remain local. Public reports contain only small
cell-level aggregates and provenance hashes. Python 3.11/3.12 package/CI checks
and before/after preservation verification are required to close the checkpoint.
