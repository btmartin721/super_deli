# Frozen tabular density/GMM comparison

Version 0.48.0.dev0; completed 2026-10-26. The [protocol](tabular-density-clustering.md)
adds an opt-in model family and preserves all previous defaults, p-values and
empirical gates. These are synthetic clustering findings, not demonstrated
species-delimitation accuracy or significance calibration.

## Completion and audits

All 300 independent observations completed their two point pipelines and the
prespecified stability subset: **3,600/3,600 successful main pipelines**, including
600 point and 3,000 subsampling runs. The independent 12-observation storage
preflight completed **624/624**. No fit failures, replacement draws or unattempted
pipelines occurred. Full local evidence is retained; no simulation arrays or raw
study inputs are published to GitHub.

The frozen scientific runtime used Python 3.12.10, NumPy 2.2.6, SciPy 1.15.3 and
scikit-learn 1.6.1. Both study arms refitted standardization and two-PC PCA. The
unchanged numerical GMM comparator used full covariance/BIC, K=1..4, minimum
size five, ten EM starts and seed 71323. HDBSCAN used the frozen parameters in
the protocol; its cluster count is unrestricted. No settings were changed after
fresh outcomes were examined.

Read-only audits verified source hashes, raw values, IDs, seed streams, all
successful PCA fits/projections/distances, selected-GMM densities/membership/BIC,
selection eligibility, paired sampling masks and all support denominators.
The underlying HDBSCAN algorithm was not independently reimplemented. A separate
post-fit audit checked contingency-table ARI, candidate sizes, exported rates,
paired metrics and exact simultaneous intervals. Neither audit refitted models.

## Recovery and coverage

Correct K is one descriptive target, not sufficient recovery. ARI below treats
each predicted noise observation as its own unassigned singleton among the
non-background generating groups. GMM assigned every observation; its ARI does
not receive a noise exemption. Background detection is evaluated separately.

| Scenario | N | HDBSCAN correct K | GMM correct K | HDBSCAN mean ARI | GMM mean ARI | HDBSCAN assigned |
|---|---:|---:|---:|---:|---:|---:|
| gaussian | 64 | 25/25 | 25/25 | 0.000 | 1.000 | 36.8% |
| gaussian | 256 | 25/25 | 25/25 | 0.000 | 1.000 | 74.1% |
| four_spherical | 64 | 20/25 | 18/25 | 0.761 | 0.911 | 97.6% |
| four_spherical | 256 | 23/25 | 25/25 | 0.956 | 0.995 | 97.3% |
| four_unequal | 64 | 24/25 | 24/25 | 0.984 | 0.996 | 99.3% |
| four_unequal | 256 | 24/25 | 25/25 | 0.994 | 1.000 | 99.5% |
| moons | 64 | 0/25 | 0/25 | 0.072 | 0.390 | 94.2% |
| moons | 256 | 4/25 | 0/25 | 0.333 | 0.446 | 97.6% |
| background | 64 | 25/25 | 12/25 | 1.000 | 0.826 | 92.1% |
| background | 256 | 22/25 | 23/25 | 0.997 | 0.859 | 91.9% |
| cline | 64 | — | — | — | — | 87.8% |
| cline | 256 | — | — | — | — | 81.7% |

In the N=64 background cell, HDBSCAN retained all generating-group observations
and recovered their partition exactly, while detecting 63% of generated background
points as noise on average. Its mean noise precision was 1.0. Some background
points overlap group supports and are absorbed into groups; these are not guaranteed
outlier labels. At N=256, HDBSCAN detected 63.4% of background on average, with
mean noise precision 0.986 and generating-group assigned coverage 99.84%.
GMM has no noise output and selected fewer than four components in 13/25 N=64
background observations.

Performance is not uniformly better. At N=256, GMM recovered four spherical and
unequal groups in all 25 observations per cell, while HDBSCAN occasionally selected
five groups. In the N=64 spherical cell, HDBSCAN selected four groups more often
but had lower mean ARI, illustrating that correct K does not guarantee correct
membership. Both methods performed poorly on noisy moons with these frozen
settings; no density-parameter tuning or winning-model promotion followed.

Both methods selected K=1 in all 50 Gaussian controls. HDBSCAN nevertheless left
63.2%/25.9% unassigned on average at N=64/256. Its ARI is zero because the one-group
truth is split between a retained density core and unassigned singleton outcomes.
Thus correct-K selection cannot be interpreted as complete population recovery.
These are algorithmic control outcomes, not species false-positive rates.

For the cline, HDBSCAN selected K>1 in 1/25 N=64 and 0/25 N=256 observations;
GMM did so in 5/25 and 21/25. A continuous cline has no discrete reference K or
truth ARI in this study. Extra components describe structure and do not establish
species or invalidate a significance test that was not run here.

## Conditional stability and uncertainty

The first five prespecified datasets per cell received 25 paired 80% subsamples
per family. All unit and pair inclusion coverage was complete across the retained
draws. HDBSCAN's mean selected-K agreement with its own point fit was 0.768/0.856
for N=64/256 spherical groups and 1.0/1.0 for the background cells. The corresponding
GMM values were 0.528/1.0 and 0.432/0.864. These compare each estimator to its own
point partition, not to a shared target or species truth. Stable wrong or selective
partitions remain possible; see the full coassignment and noise summaries.

Twenty primary correct-K proportions have simultaneous exact 95% intervals
(Bonferroni). With only 25 independent datasets/cell those intervals are broad;
for example 25/25 has a simultaneous interval of approximately 76.5–100%.
JSON/CSV exports retain all counts and exact limits. Secondary paired differences,
ARI/coverage and stability means have descriptive unadjusted approximate t
intervals over independent datasets. Five stability datasets/cell are not 125
independent observations; conditional draw-level Monte Carlo precision is separate.

## Scope and reproducibility

The measured preflight projected 110,306,971 bytes of new retained evidence,
including its allowance, below the approved 1 GiB cap. Final measured retention
and package/CI checks are recorded in local verification artifacts. Only code,
docs, invented examples and compact aggregates are published, under a 16 MiB
public-payload ceiling. All previous checkouts and study evidence were preserved;
no real empirical inputs were fitted and nobody was contacted.

Reproduce small summaries from the local audited artifact root with:

```bash
python examples/density-clustering/report.py \
  --artifact-root /path/to/local-density-study \
  --output-dir /path/to/new-small-summaries
```

Public [summary](../examples/density-clustering/results/summary.json),
[primary rates](../examples/density-clustering/results/primary-rates.csv),
[paired differences](../examples/density-clustering/results/paired-differences.json),
[coverage and agreement](../examples/density-clustering/results/metric-uncertainty.json)
and [stability](../examples/density-clustering/results/stability.json) retain the
comparison and its uncertainty. The complete simulated inputs, transformations,
labels, draw arrays and audit inventories remain local.

This checkpoint establishes a second, explicitly scoped clustering family.
It does not add genomic/shape/integrative density adapters, learned landmark
identification, deep learning, automatic model-family selection, empirical species
validation or new calibrated significance. Existing PCA/GMM defaults and all
historical p-values remain unchanged.
