# Multimodal signal-loss diagnosis: completed results

The strongest supported mechanism is loss during the final two-coordinate
projection. The per-view shape PCA preserved nearly all generating separation;
adding uninformative views and compressing the combined representation discarded
much of it. Covariance-family differences alone did not explain the shape loss.
Retaining all eight per-view coordinates helped, but did not solve every N=64 case.

## Evidence and execution

The source was the completed 0.81.0.dev0 study at
`5936eab934772e08e40d275d3d492389b9696e0f`. A checksum-bound fit-free audit
covered all 24 archived observations. The main point study reused those inputs
and generated 48 fresh independent observations: four per six-law/sample-size
cell at N=64 and N=256, with seeds 6200 through 6247. The unchanged simulator
generated 64 diploid SNPs, four traits per continuous table and six fixed 2D
landmarks; specimen IDs were matched explicitly.

All 1,080 main searches completed: 216 native workflows and 864 diagnostic
cached-score searches. The independent 72-search preflight also passed.
All 4,608 attempted component fits passed independent retained-parameter density
checks. The main and preflight native replays reproduced 2,828 archived numerical
arrays exactly. There were no withheld searches, replacements or new p-values.
The preflight projected 120.5 serial seconds and 614.2 MiB including allowances;
main execution plus its embedded audit took 62.8 seconds. Full scientific study
outputs occupied about 310 MiB before final verification evidence.

## Where the signal went

In the four archived shape-only cases, the two shape PCs retained 99.74–99.89%
of simulated group-centroid separation. After equal-view normalization and
projection from eight coordinates to two, the isolated shape contribution
retained only 14.35–38.08% (median 22.71%). Identity, preprocessing, frozen
projection and mixture-likelihood checks passed.

On the common specimen order, both full- and shared-covariance heads recovered
four shape groups in all four archived and all eight fresh shape-only datasets.
Thus changing the covariance family was insufficient to explain the loss.
Adding a nonsignal view could lose recovery even with a shared head and only
two views. The traces locate measurable loss at the integration projection;
they do not establish a universal explanation for all possible datasets.

## Fresh comparisons

The table reports correct K **and** adjusted Rand agreement of at least 0.8.
Genomic-only truth has two generating groups; shape/complementary truth has four.
All three dimension arms below use minimum cluster size nine, isolating
dimension from that feasibility rule. The legacy two-PC/minimum-five baseline
remains separately reported in the aggregate file.

| Fresh law and sample size | Two joint PCs | Four joint PCs | All eight PCs |
|---|---:|---:|---:|
| Shape-only, N=64 | 0/4 | 0/4 | 2/4 |
| Shape-only, N=256 | 0/4 | 0/4 | 4/4 |
| Genomic-only, N=64 | 2/4 | 4/4 | 4/4 |
| Genomic-only, N=256 | 4/4 | 4/4 | 4/4 |
| Complementary, N=64 | 3/4 | 3/4 | 3/4 |
| Complementary, N=256 | 4/4 | 4/4 | 4/4 |

All eight continuous-control datasets selected K=1 in these three arms.
Redundant-signal recovery was 8/8 throughout. Conflicting-view simulations had
no privileged species partition and were not scored as four-species recovery.
The higher-dimensional representation retained all fused-coordinate distances
and centroid separation numerically; the two remaining N=64 shape failures
show that retaining information does not guarantee mixture recovery.

Across the two fresh sample sizes, shape recovery rose from 0/8 to 6/8.
Uncertainty is assessed separately for each fixed sample size: the unadjusted
exact 95% interval is approximately 7–93% for 2/4 at N=64, and 40–100% for
4/4 at N=256. The corresponding 0/4 two-PC interval is 0–60% in each cell.
No common recovery probability is assumed across the two sample sizes. These
tiny cells support a descriptive development decision, not general false-positive
or power guarantees. Detailed per-cell intervals and explicit failure bounds
are in the [aggregate summary](multimodal-signal-summary.json). Repeated arms on
one dataset are paired observations, not independent replicates.

## Frozen decision and limits

The prespecified follow-up gate passed: all four archived shape cases retained
less than half of shape separation at two joint PCs; eight PCs preserved full
energy; fresh correct shape recovery gained six cases; there were no additional
control splits, complementary recovery losses or failures.

The next proposed checkpoint is an **opt-in fusion policy retaining all selected
per-view PCs**, with dimension-appropriate native feasibility checks, frozen
projection and independent fresh validation. This checkpoint does not implement
that production option, promote a new default or authorize choosing dimensions
after looking at labels. Its significance calibration and resampling behavior
would require their own stated validation boundaries.

No neural training, calibration, bootstrap, empirical analysis or author contact
occurred. Historical p-values and empirical readiness gates are unchanged.
Cached-score contrasts hold per-view transformations fixed and are not new
full-pipeline significance tests. BIC gains/penalties were audited within each
score space; absolute BIC values across dimensions were not ranked. High-
dimensional in-sample linear recovery is not an information bound. Full raw
inputs, identities, simulated labels and fitted parameters remain local.
