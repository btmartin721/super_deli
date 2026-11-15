# Truth-free initialization: completed paired validation

The new diverse_v1 policy is an experimental opt-in for full-retention matched
shared-GMM analyses. It improves fresh N=64 shape-only point recovery over ten
starts, but does not establish an advantage over ordinary 100 kmeans starts and
does not resolve bootstrap splitting of single-population controls. No model
default or historical p-value changes. See the [interface](truth-free-init.md).

## Frozen design and full-pipeline checks

The main study generated 200 fresh independent datasets:twenty per law/N cell
across shape-only, genomic-only, complementary, continuous single-population
control and conflicting-view stress at N=64/256. Seeds 8100..8299 and all settings
were fixed before generation. Every observation has 64 diploid biallelic SNPs,
four continuous traits per morphology/ecology view and six fixed 2D landmarks.
Source row orders differ and must align by declared specimen IDs.

The unchanged simulator draws genotypes from independent Binomial(2,p) loci,
continuous traits from Gaussian noise with declared group shifts where present,
and landmark coordinates from a common outline plus Gaussian perturbations and
prespecified shape modes. Shape-only has four generating groups; genomic-only
has two; complementary combines genomic and shape axes into four. Control
genotypes share pooled frequencies, traits have no group shifts and shape has no
group modes. Its processed eight-coordinate distribution is not an exact
Gaussian-score null. Conflicting views deliberately have no unique joint target;
their partitions/stability are descriptive, with no recovery rate assigned.
There are no empirical datasets or inferred species truths in this study.

Compare unchanged ten starts, unchanged 100 kmeans starts, and one fixed bank of
34 kmeans/33 k-means++/33 random_from_data starts. All arms use two PCs/view, all
eight joint coordinates, equal weights, shared covariance, BIC,K=1..4, minimum
nine distinct original individuals, model seed 301 and unchanged numerical
tolerances. Each K receives the same arm-specific start budget. Batch selection
precedes the unchanged size check; no alternate eligible start is substituted.

Every native workflow reads its original sources, including SNPio ingestion.
The first three fresh observations/cell have 25 paired whole-specimen draws/arm:
2,250 main bootstrap refits. Each draw resamples raw ingested arrays and refits
preprocessing,GPA,view PCA,normalization,fusion,joint PCA and the complete K
search. It does not rewrite/re-read genomic source files. Original loci and
identities stay fixed; multiplicities and distinct-unit counts remain separate.

The 2,880 main searches comprise 600 fresh points,2,250 draws and 30 archived points.
Those archived points include 18 exact legacy replays and twelve new-policy
comparisons, excluded from fresh performance estimates. A separate ten-dataset
preflight completed 785 searches, including 750 draws and three exact old replays.
All 3,665 pipeline refits completed with zero withheld points or failed draws.
Independent audits checked 24,352 captured mixture batches,3,665 raw shape refits,
full-space geometry/projections, paired identities and OOB denominators. The 21
legacy replays matched 1,344 numerical arrays exactly. No new truth-initialized
fits, significance calibration, empirical fitting or neural-training study ran.

## Fresh point results

Recovery means correct K and adjusted Rand agreement of at least 0.8. Counts are
independent datasets within fixed-N/law cells; all three policies are paired.

| N | Event | Ten starts | Ordinary 100 | Diverse 100 |
|---:|---|---:|---:|---:|
| 64 | shape only recovery | 11/20 | 20/20 | 19/20 |
| 64 | genomic only recovery | 20/20 | 20/20 | 20/20 |
| 64 | complementary recovery | 19/20 | 19/20 | 19/20 |
| 64 | Control splits (K>1) | 0/20 | 1/20 | 1/20 |
| 256 | shape only recovery | 20/20 | 20/20 | 20/20 |
| 256 | genomic only recovery | 20/20 | 20/20 | 20/20 |
| 256 | complementary recovery | 20/20 | 20/20 | 20/20 |
| 256 | Control splits (K>1) | 0/20 | 0/20 | 0/20 |

For N=64 shape recovery, exact unadjusted 95% intervals are 31.5–76.9% for 11/20,
83.2–100% for 20/20 and 75.1–99.9% for 19/20. A 0/20 control-split count has upper
endpoint 16.8%;1/20 has interval approximately 0.13–24.9%. These small samples do
not establish a generally low false-split rate or superiority/equivalence between
policies. Cells are not pooled into one common binomial rate. The
[complete aggregate](truth-free-init-summary.json) retains all cells, selected K,
ARI, failure bounds and paired discordance counts with conservative per-contrast
intervals; intervals are not simultaneous across every comparison.

The remaining N=64 diverse shape miss has an undersized winning K=4 candidate.
Ordinary 100 starts recovers all twenty shape cases in this fresh cell. Thus the
gain over ten starts chiefly supports adequate optimization effort; the new
mixture of starting methods has no demonstrated recovery advantage over 100
ordinary starts. One complementary N=64 observation remains unrecovered by all
three arms. Conflicting-view points select K=4 throughout, but this is not
reported as species or target recovery.

## Bootstrap behavior remains unresolved

Each cell has only three independent parent datasets. Their 75 draws are repeated
conditional measurements, not 75 independent biological samples, false-positive
trials, species probabilities or significance tests.

| N | Conditional event | Ten starts | Ordinary 100 | Diverse 100 |
|---:|---|---:|---:|---:|
| 64 | Shape-only K=4 draws | 6/75 | 16/75 | 11/75 |
| 64 | Control split draws | 37/75 | 41/75 | 34/75 |
| 256 | Shape-only K=4 draws | 71/75 | 75/75 | 75/75 |
| 256 | Control split draws | 33/75 | 43/75 | 40/75 |

At N=64, shape-only K=4 is eligible in 6/75,16/75 and 11/75 draws respectively.
The other exclusions are 66/56/61 undersized winning candidates and three draws
per arm with fewer than 36 distinct training individuals. Draws contain 34–46
distinct originals despite retaining 64 weighted rows. The eligibility pattern
does not by itself distinguish sampling-information loss from optimization that
finds an undersized solution. The existing best-fit-before-size rule is retained.

Median parent-level exact own-point-cluster recovery for N=64 shape is 0.04,
0.24 and 0.12; at N=256 it is 0.96,1.00 and 1.00. These measure agreement with
each policy's point partition, which can itself be wrong. Common ten-start-point
recovery and OOB coverage are retained separately in the aggregate. OOB pair
coverage is about 97%, measuring available comparisons rather than correctness.

Control splitting persists at both sample sizes. The diverse bank has fewer
split draws than ten starts at N=64 and more at N=256, with only three parents
per cell. No consistent resampling correction or reliable species count is
established. These outcomes must not be interpreted as calibrated p-values.

## Numerical guards, retained failures and resources

The first preflight audit incorrectly assumed a joint recovery target for
conflicting views, although native metrics correctly retained null values. That
audit/report accounting was repaired and all 785 existing searches re-audited,
with no refits or changes to scientific settings. The failed audit is preserved.

The first main observation then exposed a hard-label disagreement in an unused
initialization batch with almost identical component probabilities. The failed
guard and all 78 completed refits were preserved. Under the explicitly approved
audit-only amendment, probability/likelihood/BIC tolerances remain unchanged for
every batch; saved labels must match saved-posterior argmax for every batch;
independent hard labels must agree exactly for every winning batch at every K
and every legacy fit. Only unused-batch label ambiguity is recorded separately.
Across the complete study,14,600 used-batch label checks passed; four unused
batches had 39 ambiguous rows. No unique-label claim is made for those unused
batches, and none can affect eligibility or reported assignments.

All original scientific arrays remained unchanged. Exactly 2,802 remaining main
searches resumed after re-auditing the 785 preflight and 78 retained main searches.
There were no replacement datasets, starts or draws, and no fitted model or
default changed in the amendment. Strict audit defaults remain intact outside
the explicit unused-initialization scope.

The preflight projected about 3.8 GiB under the eight-GiB local evidence cap and
eight-GiB free reserve. Main execution/audit, including charged repair work,
took about 15.5 serial minutes under one scientific single-thread worker, within
the one-hour bound. Point workflow medians include ingestion, export and audit:
approximately 0.117/0.425/0.304 seconds at N=64 and 0.264/0.629/0.485 seconds at
N=256 for ten/ordinary 100/diverse 100. These are local workflow timings, not
isolated optimizer costs or general hardware/runtime guarantees.

## Decision and limitations

Retain diverse_v1 as an experimental opt-in. Promote no optimizer, start count,
model or default automatically. Ordinary 100 starts matches or exceeds its shape
recovery in these cells; neither 100-start policy resolves control/stability
limitations. Further unguided initialization searches are not justified by this
bounded evidence. Persistent bootstrap splitting and distinct-member eligibility
are the next reliability questions; any resampling/selection correction needs
its own prespecified validation.

This study supplies no full-retention significance calibration or species-boundary
guarantee. P-values and empirical evidence gates stay unchanged. Heavy tails,
unequal spreads, spatial dependence, linked loci, missing views, grouped sampling
and real biological species boundaries remain outside this bank's validation.
Full retention preserves selected per-view PCs, not every raw feature.

All detailed inputs, parameters, failed guards, approvals, source seals and
resampling traces remain local. Only code, invented fixtures and small aggregates
are published. No author contact occurred.
