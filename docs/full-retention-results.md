# Full-retention fusion: completed validation and N=64 audit

The opt-in retain_all_v1 preset is available through the native API and the
fit-free full-retention-config CLI. It keeps every selected per-view PC using
the unchanged full-rank joint PCA/GMM path. The ordinary two-PC/full-covariance
defaults, ten-start optimization and historical p-values are preserved.

## Scope and audit evidence

The main study generated 160 fresh independent matched datasets: twenty per
law/sample-size cell across shape-only, genomic-only, complementary and
continuous-control laws at N=64/256, with frozen seeds 7200 through 7359.
Each contained 64 diploid SNPs ingested through SNPio, four continuous traits per
morphology/ecology view and six fixed 2D landmarks. Each native workflow ingested
its source files before fitting. Native draws resampled raw arrays before all
numerical preprocessing; SNPio was not re-run for every bootstrap draw.

The three prespecified shared-covariance BIC arms were two PCs/minimum five,
two PCs/minimum nine, and all eight PCs/minimum nine. The last two isolate the
dimension change from the existing size guard. All used two PCs/view, equal view
weights, K=1..4, model seed 301 and ten starts. The first three datasets/cell had
25 paired whole-specimen draws per arm: 1,800 main bootstrap refits. Source loci,
identities, draw multiplicities and OOB denominators remained explicit.

All 2,332 main searches and 22 separate truth-initialized component fits completed.
There were no withheld points, failed draws, replacement observations or new
scientific p-values. A separate preflight completed 637 searches, including
600 bootstrap refits, and two truth-initialized fits. The main audit verified
9,304 fitted native candidates, 800 full-coordinate distance checks, 2,332 raw
shape refits and 3,056 exact replayed numerical arrays. The preflight independently
verified another 2,540 native candidates and 219 full-coordinate checks.
Truth-informed diagnostics remained separate from native selection throughout.

The main run plus its embedded/retrospective audits took 244.3 serial seconds
under the frozen single-thread settings. Full main evidence occupied about
1.54 GiB before final verification artifacts. The preflight's conservative
4.33-GiB projection exceeded an initial agent-defined 4-GiB budget. That failed
gate and original configuration were preserved; a storage-only amendment raised
the local cap to 6 GiB, with eight GiB free reserved. The same completed preflight
fits were re-audited and the gate recomputed, adding zero scientific fits and
changing no seeds, scientific settings, thresholds or model defaults.

## Fresh point recovery

Recovery means correct K **and** adjusted Rand agreement of at least 0.8.
The genomic-only law has two generating groups; shape-only and complementary
laws have four. Controls have no privileged four-species truth; their event is
selection of K greater than one. Counts below are independent datasets, with
paired model arms on each dataset.

| N | Event | Two PCs/minimum 5 | Two PCs/minimum 9 | All eight/minimum 9 |
|---:|---|---:|---:|---:|
| 64 | Shape-only recovery | 0/20 | 0/20 | 11/20 |
| 64 | Genomic-only recovery | 19/20 | 20/20 | 20/20 |
| 64 | Complementary recovery | 20/20 | 20/20 | 20/20 |
| 64 | Continuous-control splits | 2/20 | 1/20 | 0/20 |
| 256 | Shape-only recovery | 0/20 | 0/20 | 19/20 |
| 256 | Genomic-only recovery | 20/20 | 20/20 | 20/20 |
| 256 | Complementary recovery | 20/20 | 20/20 | 20/20 |
| 256 | Continuous-control splits | 0/20 | 0/20 | 0/20 |

For full-retention shape recovery, the unadjusted exact 95% interval is 31.5–76.9%
for 11/20 at N=64 and 75.1–99.9% for 19/20 at N=256. Each 0/20 control-split
count has an upper endpoint of 16.8%; zero observed splits is not a general
false-split guarantee. Intervals are calculated within fixed-N/law cells and
are not simultaneous across all comparisons. No common binomial recovery rate
is assumed across sample sizes. The [complete aggregate](full-retention-summary.json)
includes all cells, paired outcome counts, selected K, ARI and failure bounds.

## What the N=64 audit established

The point audit included all twenty fresh N=64 shape-only datasets and the two
archived under-recovery cases. The unchanged ten-start workflow recovered 11/20
fresh datasets. One prespecified 100-start comparison recovered 19/20, correcting
eight of the nine misses without losing a previously correct case. Both archived
misses also recovered four groups with 100 starts. The fitted score arrays were
exactly identical between ten- and 100-start comparisons.

All nine fresh ten-start misses had an eligible, lower-BIC K=4 solution after
truth-informed initialization. Their BIC improvements over the native selected
model ranged from approximately 90 to 132 points in the **same** score space.
Three native K=4 candidates had undersized components; the other six were eligible
but lost on BIC. Thus a BIC penalty alone cannot explain these observed misses:
the optimizer did not find those better admissible solutions. This does not
establish a global optimum or turn truth-informed fitting into a deployable method.

One fresh case still selected K=2 with 100 starts even though its truth-initialized
K=4 solution was eligible and had much lower BIC. We retained the existing rule
that fits each K first and then checks component size; no alternate-start
admissibility or automatic fallback was introduced. Neither 100-start nor truth-
initialized fits replaced the deployed ten-start results. Control and stability
performance of a future optimization change requires separate validation.

## Conditional bootstrap stability remains a limitation

Bootstrap evidence came from three independent parent datasets per cell, with
25 draws each. These 75 draws are repeated measurements, not 75 independent
biological datasets or false-positive trials. All raw preprocessing, GPA, view
PCA, normalization, joint rotation and complete GMM searches were refitted.

For N=64 shape-only full retention, bootstrap K counts across the three parents
were K=2 in 65 draws, K=3 in five and K=4 in five. The median parent-level mean
exact point-cluster recovery was 0.08. At N=256 the corresponding K=4 count was
65/75 and median exact point-cluster recovery was 0.84. Reproducing a point
partition is distinct from recovering generating groups; a reproducible merge
can still be wrong.

Although all forty fresh full-retention control points selected K=1, their
bootstrap draws split into multiple components in 47/75 N=64 draws and 35/75
N=256 draws. These conditional resampling outcomes show substantial instability.
The full-retention option therefore does not provide a generally reliable
bootstrap species count. Both sampling variation and fixed optimizer/size rules
remain part of the conditional procedure; the point optimization audit does not
isolate their contributions to bootstrap behavior.

Requested OOB pair coverage was about 97% at both sample sizes, with independently
verified pair-specific counts. Coverage measures which pairs were omitted, not
the accuracy of their assignments. No stability draws failed in this study.

## Deployment boundary and next research question

The new policy is opt-in and restricted to complete, explicitly matched specimen
views with prespecified shared covariance/BIC. The preset, frozen projection,
native unit bootstrap, metadata and unified report are tested; numerical fitters
and legacy configuration bytes remain unchanged. Significance workflows reject
the tagged policy pending separate calibration. Existing explicitly configured
higher-PC legacy workflows retain their former behavior.

The evidence supports retaining the option as experimental. The next targeted
question is whether prespecified **unsupervised initialization** can find the
better solutions consistently without worsening control splitting or bootstrap
behavior. This checkpoint implements no optimizer change, chooses no winning
seed and promotes no new default. Higher K ranges, unequal spreads, heavy tails,
spatial dependence, missing views, grouped sampling and empirical species-boundary
validation remain outside this bounded study. Empirical evidence gates remain
closed. No neural-training study, significance calibration or author contact occurred.

Full raw inputs, fitted parameters, source seals, failures, resource amendments
and independent audit traces remain local. Only code, invented tests and small
aggregate outputs are published.
