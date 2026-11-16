# Whole-individual 80% subsampling: results

This checkpoint adds a separate opt-in sensitivity analysis alongside an unchanged native point run. Existing numerical models, preprocessing, minimum-size rules, defaults, assignments, schemas and historical p-values remain unchanged. Detailed evidence stays local.

The frozen main study used 48 fresh independent datasets: eight per N/law cell, N=64/256, with complete matched genomic, morphology, individual ecology and fixed-landmark shape views. Each dataset supplied one ordinary 100-start full-retention/shared-GMM point fit, 25 native whole-individual bootstrap refits and 25 80% without-replacement refits. Optimizer seeds are paired; sampling streams differ. An independent 1,024-individual evaluation cohort is never used for fitting or selection. Four exact archived 26-search replays bring the main budget to 2,552 searches; a separate six-dataset/two-replay preflight used 358.

The one-population law uses unlinked diploid binomial SNPs, independent Gaussian traits and noisy fixed landmarks. It is a control for component splitting, not a claim that every transformed coordinate is Gaussian. Four-group laws add two orthogonal shape directions with equal within-group coordinate noise; all other views remain non-discriminatory. Balanced counts are 16/16/16/16 or 64/64/64/64. Unequal counts are 25/16/13/10 or 100/64/52/40. Generating labels are used only after fitting. These laws do not span natural species histories, linkage, missingness or empirical dependence.

## Recovery and control splitting

For controls the event is selected K>1. For four-group data, recovery requires K=4 and adjusted Rand agreement at least 0.8 with the generating labels. Point counts use eight independent datasets per row. Draw counts are 200 conditional repeated measurements from those eight datasets, not 200 independent observations.

| N | Law | Point event | Bootstrap event | 80% event | Bootstrap K=4 eligible | 80% K=4 eligible |
|---:|---|---:|---:|---:|---:|---:|
| 64 | One-population control | 1/8 | 96/200 | 17/200 | 2/200 | 50/200 |
| 64 | Balanced four shape groups | 8/8 | 30/200 | 182/200 | 32/200 | 189/200 |
| 64 | Unequal four shape groups | 8/8 | 1/200 | 64/200 | 1/200 | 68/200 |
| 256 | One-population control | 0/8 | 107/200 | 0/200 | 60/200 | 188/200 |
| 256 | Balanced four shape groups | 8/8 | 200/200 | 200/200 | 200/200 | 200/200 |
| 256 | Unequal four shape groups | 8/8 | 200/200 | 200/200 | 200/200 | 200/200 |

Paired parent-level event differences (80% minus bootstrap) and marginal descriptive intervals are retained in the small JSON summary. These are conditional sensitivity comparisons under fixed sampling/optimizer streams. Eight independent datasets per cell support an operating-range description, not a general error guarantee or automatic policy promotion. Original-point event rates also include exact binomial intervals.

## Sampling support and coverage

Every 80% draw has 51 distinct original individuals at N=64 or 204 at N=256. The existing minimum remains nine distinct individuals per fitted component. Bootstrap multiplicities and their lower distinct support remain unchanged. The new report measures actual representation of each original candidate cluster, separately from what the refitted GMM selected. A point partition can itself be wrong; agreement with it is not recovery of true biological groups.

| N | Law | Mean bootstrap distinct individuals | Bootstrap draws with a point cluster below nine | 80% draws with a point cluster below nine | 80% omitted-pair coverage |
|---:|---|---:|---:|---:|---:|
| 64 | One-population control | 40.80 | 8/200 | 0/200 | 63.7% |
| 64 | Balanced four shape groups | 40.80 | 117/200 | 1/200 | 63.7% |
| 64 | Unequal four shape groups | 40.80 | 197/200 | 128/200 | 63.7% |
| 256 | One-population control | 161.12 | 0/200 | 0/200 | 65.0% |
| 256 | Balanced four shape groups | 161.12 | 0/200 | 0/200 | 65.0% |
| 256 | Unequal four shape groups | 161.12 | 0/200 | 0/200 | 65.0% |

Included-pair coassignment is conditional on both individuals being sampled in a successful refit. Omitted-pair coassignment uses frozen projections only when both were left out; its requested/successful denominators and coverage are separate. Only about 4% of pairs are jointly omitted in a particular 80% draw, so 25 draws cannot cover all pairs. No omitted comparison, or an affected failed omitted evaluation, remains unavailable. Failed-outcome identification bounds are not confidence intervals.

## Independent prediction and limitations

| N | Law | Bootstrap K>1 without positive external gain | 80% K>1 without positive external gain | Withheld bootstrap / 80% searches |
|---:|---|---:|---:|---:|
| 64 | One-population control | 96 | 17 | 0 / 0 |
| 64 | Balanced four shape groups | 6 | 0 | 0 / 0 |
| 64 | Unequal four shape groups | 11 | 0 | 0 / 0 |
| 256 | One-population control | 103 | 0 | 0 / 0 |
| 256 | Balanced four shape groups | 0 | 0 | 0 / 0 |
| 256 | Unequal four shape groups | 0 | 0 | 0 / 0 |

External gain compares the selected candidate with K=1 within the same frozen score space. It assesses independent predictive fit under the declared law; it is not species evidence. The two resampling methods change training size, distinct support, weights, preprocessing and BIC together. Their difference cannot be attributed to one mechanism alone. No new model, p-value, calibration, empirical fit, neural training or author contact is introduced.

The command is deliberately opt-in. It retains cluster representation, candidate eligibility, exact included recovery, Jaccard similarity, splitting/merging, failure bounds, and included/omitted coverage so small-group risks remain visible. Improved sensitivity behavior on one law cannot guarantee recovery of rare groups, fix model misspecification or establish species boundaries.

## Verification

All 2,910 main/preflight searches were independently re-audited without refitting. Exact archived numerical parity covers 8,100 main arrays. The public summary records 0 withheld main diagnostic searches; failures are retained without replacements. Engineering checks passed 2,740 scientific and 1,863 foundation tests. Wheel/sdist, installed new/legacy commands and Python 3.11/3.12 CI are required before completion is sealed.

The original tuple-versus-JSON-list metadata-audit failure is retained locally. The controller repair normalized only declaration representation; it changed no scientific source, array, setting, seed, fit or tolerance. Detailed raw inputs, fitted parameters and projections remain local. Only code, invented fixtures, documentation and small aggregates are published.
