# Held-out predictive support and rare-group feasibility: results

All frozen descriptive targets passed. The diagnostic remains experimental, with no filter or default promotion. No descriptive target failed.

This checkpoint adds a separate opt-in predictive report to an existing native
80% whole-individual subsampling run. It changes no original partition, numerical
default, historical schema or p-value. Positive predictive gain can support some
structure without validating a particular K, every extra cluster or species boundaries.

## Frozen design and denominators

The main study used 200 fresh independent datasets, 20 per cell below, with 64
independent diploid biallelic SNPs, four Gaussian traits in each of two continuous
views, and six planar fixed landmarks. Only shape carried group signal in the
four-group laws. Landmark noise SD was .025 and the two shape modes had amplitude
.18. These simplified laws do not represent natural species histories, linkage,
missingness or empirical dependence. Each parent supplied an independent
1,024-individual reference cohort, excluded from fitting and model selection.

Each dataset used one original point search and 25 native 80% subsample searches,
followed by 25 native K=1 score-head replays. Model seed 301, sampling seed 1601,
two PCs per view, eight joint PCs, equal weights, tied covariance, BIC K=1..4,
minimum nine distinct individuals and ordinary 100 starts were fixed in advance.
Main parameter seeds were 21000..21199; observation and evaluation innovations
were the two children of SeedSequence([parameter_seed, 20261117]). Truth labels
entered evaluation only, without stratified sampling or outcome-based tuning.

Four exact legacy subsample replays bring the main budget to 5,300 complete
pipeline searches and 5,100 K=1 head searches: 10,400 total. The separate preflight
used 610 searches. A fit-free audit covered all 48 preceding datasets and 1,200
retained draws. Completed archive/preflight fits were not repeated for the storage
amendment.

## Primary events

Every denominator is 20 independent parents. Qualification requires at least 95%
individual coverage and no failed requested predictive comparison; 95% is a
reporting policy, not a confidence level. A qualified positive event requires
positive internal gain and a complete external reference. Direction agreement
also requires complete qualified evidence. Joint recovery requires the correct
point K, ARI >= .8, smallest-true-group Jaccard >= .8, and the appropriate internal
gain direction: nonpositive for controls and positive for alternatives.

| Cell | True counts | Qualified | Qualified positive | Direction agreement | Joint recovery/direction | Correct point recovery |
|---|---|---:|---:|---:|---:|---:|
| N=64, control | 64 | 20/20 | 1/20 | 19/20 | 16/20 | 17/20 |
| N=128, control | 128 | 20/20 | 0/20 | 20/20 | 20/20 | 20/20 |
| N=256, control | 256 | 20/20 | 0/20 | 20/20 | 20/20 | 20/20 |
| N=64, balanced | 16/16/16/16 | 20/20 | 20/20 | 20/20 | 18/20 | 18/20 |
| N=64, proportional | 25/16/13/10 | 20/20 | 20/20 | 20/20 | 19/20 | 19/20 |
| N=64, rare six | 29/16/13/6 | 20/20 | 20/20 | 20/20 | 0/20 | 0/20 |
| N=128, proportional | 50/32/26/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |
| N=256, proportional | 100/64/52/40 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |
| N=128, fixed ten | 55/35/28/10 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |
| N=256, fixed ten | 114/73/59/10 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |

The [primary summary](heldout-support-summary.json) retains all 40 rates, unavailable
counts, bounded-event identification endpoints, exact marginal 95% intervals and
simultaneous exact intervals using a Bonferroni family of 40. Twenty parents per
cell provide limited precision. Repeated omissions and draws are conditional
measurements, not independent replicates or additional interval denominators.

The frozen targets were qualification >=90% in every cell, qualified positive
<=10% in each control, direction agreement >=80% in every cell, and qualified
positive >=80% in the balanced and proportional four-group cells. Desired-event
targets use lower identification bounds; control-positive limits use upper bounds.
The six-member and fixed-ten cases are feasibility stress cases, without an
additional recovery target. No threshold, seed, model or target was changed.

## Predictive gains

Internal gain averages available omitted comparisons per individual first, then
weights covered individuals equally. The external reference averages each draw's
mean gain over the independent cohort. These averaging schemes are distinct.
Both compare selected versus K=1 log predictive density in each draw's frozen
projected score space, rather than raw-data likelihood. Training transformations
remain frozen for all omitted and external projections.

| Cell | Median internal gain | Median external gain | Median point ARI |
|---|---:|---:|---:|
| N=64, control | -0.062 | -0.064 | 1.000 |
| N=128, control | 0.000 | 0.000 | 1.000 |
| N=256, control | 0.000 | 0.000 | 1.000 |
| N=64, balanced | 1.993 | 2.033 | 1.000 |
| N=64, proportional | 1.468 | 1.389 | 1.000 |
| N=64, rare six | 1.052 | 1.061 | 0.545 |
| N=128, proportional | 2.444 | 2.398 | 1.000 |
| N=256, proportional | 2.515 | 2.521 | 1.000 |
| N=128, fixed ten | 1.619 | 1.678 | 1.000 |
| N=256, fixed ten | 1.853 | 1.922 | 1.000 |

There were 0 numerical-withheld main searches and
0 unavailable predictive draws among the fresh parents. Failed or
uncovered comparisons remain unavailable; they are not converted to K=1 or
negative evidence. Missing continuous gains are unbounded, so no finite
failed-outcome bounds or pseudo-replicate confidence intervals are supplied.
The public JSON retains availability separately from descriptive gain summaries.

## Smallest-group recovery and sampling support

The smallest generating group is matched by largest Jaccard, with deterministic
label-order ties. Point Jaccard, precision and recall below are medians over
available parents. Native-draw recovery uses saved frozen projections for all
original individuals and the same correct-K/ARI/Jaccard predicate as point
recovery. It is distinct from agreement with the original fitted partition.

| Cell | Smallest group | Point Jaccard | Point precision | Point recall | Native projected recovery | K=4 eligible | True smallest group below nine | Original-point agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N=64, balanced | 16 | 1.000 | 1.000 | 1.000 | 428/500 | 456/500 | 0/500 | 0.939 |
| N=64, proportional | 10 | 1.000 | 1.000 | 1.000 | 140/500 | 154/500 | 326/500 | 0.637 |
| N=64, rare six | 6 | 0.294 | 0.294 | 1.000 | 0/500 | 13/500 | 500/500 | 0.792 |
| N=128, proportional | 20 | 1.000 | 1.000 | 1.000 | 499/500 | 500/500 | 0/500 | 1.000 |
| N=256, proportional | 40 | 1.000 | 1.000 | 1.000 | 500/500 | 500/500 | 0/500 | 1.000 |
| N=128, fixed ten | 10 | 1.000 | 1.000 | 1.000 | 172/500 | 175/500 | 327/500 | 0.911 |
| N=256, fixed ten | 10 | 1.000 | 1.000 | 1.000 | 180/500 | 188/500 | 320/500 | 0.972 |

There are 500 requested draws from 20 parents per row. Known native recovery
counts use the requested denominator; candidate eligibility uses complete native
draws and representation uses draws with known sampling exposure. Unavailable
draws remain unknown, with bounded recovery endpoints in the small summary. Draws describe
conditional sensitivity, not 500 independent observations. Original-point agreement
is the median of each parent's mean ARI over complete draws. The
[minority summary](heldout-minority-summary.json) also retains parent-level recovery
and failure-bound summaries. Truth-based representation is evaluation-only;
production reports annotate original point clusters, which can miss a true rare
group entirely. Positive global prediction and stability of a fitted partition
cannot establish independent confirmation of that group.

Proportional designs increase the smallest group from 10 to 20 to 40 while total
N grows from 64 to 128 to 256. Fixed-ten designs keep its abundance at ten as total
N increases. More total individuals do not by themselves guarantee that a native
80% training draw contains nine members of the smallest group. Candidate eligibility
and representation remain separate, and neither is treated as proof of recovery.
Perfectly isolating a six-member group is incompatible with the unchanged
nine-distinct-individual minimum for an eligible fitted component. That stress
case exposes an explicit feasibility limit rather than motivating a changed rule.

## Verification and retained evidence

All 10,400 attempted main searches were independently
re-audited without refitting: 7,243,925 density checks over
168,909 files and 400 known-raw cohorts.
The four legacy replays matched 14,964 existing numerical arrays. The original
610-search preflight passed 429,265 independent density checks.
The approved storage amendment verified every one of 11,292 retained scientific
files, preserved the original failed gates, and changed no scientific byte or fit.

Both amended gates used the same original preflight measurements. The storage
projection remained 11,532,263,412 bytes (10.740 GiB); only the approved
local cap increased to 12 GiB. The 25% allowance, 512 MiB reserve, >=8 GiB free floor,
single numerical thread, one scientific worker and two-hour main/audit cap remained
fixed. Charged main execution/re-audit time was
64.91 minutes before the additional fit-free
report checks, which are retained separately and charged in the completion receipt.

Amended engineering suites passed 2,750 science and 1,867 foundation tests.
Wheel/sdist allowlists, installed new/legacy workflows, Python 3.11/3.12 and all
six exact-master CI jobs are required before local completion is sealed. Existing
numerical source, defaults, historical p-values, prior worktrees and evidence are
verified separately. Detailed raw inputs, parameters, identities, predictions and
failure records remain local. Only code, invented fixtures, documentation and small
aggregates are public. No empirical analysis, new significance test or author contact
was introduced.
