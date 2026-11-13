# Frozen known-reference-only held-out-group validation

The bounded study compares fixed class-specific Gaussian reference support with
unchanged unscreened, probability, pooled-distance and fifth-neighbor policies.
It is a new three-reference experiment, not a paired improvement over preceding
four-reference studies that used different training sizes and information.

## Independent datasets and excluded groups

Freeze eighty fresh parent datasets, ten per cell across weak linear, standard
linear, allometry+saturation and confounded-size measurements at original
four-group panel N=64/256. Retain all four possible group exclusions per parent.
Each fold refits preprocessing, the unchanged logistic model and residual seed 0,
all support parameters and independent calibration on only the three remaining
groups. Actual known training N=48/192, with 16/64 per reference. Calibration
retains 120 known rows, forty/reference. No held-out training or calibration row
enters any fitting stage. One fixed neural seed is a bounded comparator, not a
seed-selection or general architecture study.

Independent parent roles: training N, calibration 160, IID 2048, nearby 512,
distant 512 and an ordinary IID overlap pool of 2048. Confounded-size adds an
independent shifted-known pool of 2048. The new seed domain is `[20261112,760,...]`.
Every role/innovation and settings/source/runtime declaration is materialized
before scientific generation. The generator's raw morphology/ecology, latent
coordinates, size, labels, row order and role maps are retained locally. Only
measured morphology/ecology and permitted known reference labels reach fitting.

Evaluation per fold uses 1536 known IID, 512 held-out group, 512 nearby and 512
distant rows. The exact-overlap stress cohort selects 512 rows from the lowest
remaining known original group in the independently generated overlap pool; it
has that known reference's measured distribution. Compare its acceptance with
that matching reference's coverage. Confounded-size also evaluates 1536
shifted-known rows. Full unused pool rows and exclusions remain accounted for.
Overlap and shift are separate stress results, outside the primary joint target.

No unfamiliar example enters training or calibration. No evaluation threshold,
classifier seed, covariance family or rule is chosen after outcomes. The .05
known-only rank setting and .1 class covariance shrinkage are fixed. Classwise
known inclusion is distinct from accepted accuracy and unknown rejection.

## Fixed workload and inference

Main: eighty parent jobs x four excluded groups x two heads =640 classifier fits.
Four exact whole-fold fit replays add eight fits. Main total: 84 jobs, 648 fits,
324 residual trainings and 162,000 actual optimizer updates. Replay cells are the
first weak-N64/held0, standard-N256/held1, combined-N64/held2 and
confounded-N256/held3 datasets. They add no independent performance observations.

Separate preflight: one fresh parent in each of eight cells, all four exclusions
and both heads, plus weak-N64/held0 and combined-N256/held3 replays. Ten jobs,
68 fits, 34 residual trainings and 17,000 updates. Across phases: 716 classifier
fits, 358 residual/358 logistic, 179,000 independently audited actual updates;
550 parent cohorts, 523,776 rows and 2,750 disjoint innovation paths. Preflight is
excluded from performance estimates. No new architecture, clustering/GMM search,
bootstrap, significance/null draw, empirical fitting or author contact.

A fold meets the extended joint target only with known coverage>=80%,
accepted-known accuracy>=90%, and acceptance<=10% separately for the held-out,
nearby and distant groups. There is one fixed known-only cutoff per class, with
no qualifying-point search. Valid withholding/ambiguity and target failure are
separate from numerical unavailability.

Parent success requires at least three of its four exclusions to meet that full
target. Four folds are repeated measurements of one parent. The primary 64 rates
span eight cells x two heads x four screened policies, each with ten independent
parents. Report exact marginal and separate Bonferroni-64 simultaneous 95%
intervals and requested-denominator failure bounds. A partial parent is a known
success/failure only when its resolved folds logically determine that outcome;
otherwise it remains unavailable. Per-excluded-group and unscreened outcomes are
secondary and retain marginal intervals only, outside the 64-rate family.

Report 48 paired new-versus-existing-policy contrasts and eight paired
new-policy neural-versus-logistic contrasts, with dataset-level discordances,
conservative exact-binomial union-bound intervals, simultaneous uncertainty over
56 contrasts and numerical failure bounds. The 64-rate and 56-contrast simultaneous intervals
are separate prespecified families. Report parent-weighted continuous
coverage/error/unfamiliar acceptance and classwise ambiguity/conflict results.
Continuous means use available folds per parent and retain availability counts;
primary outcomes keep all four requested folds and logical failure bounds.
Neither rows, folds, heads nor policies increase the independent dataset count.

The sole focal allocation flag is combined measurements, original N=256,
residual seed zero: at least five of ten parents meet the three-of-four target,
at least three succeed only with new support versus the local policy, at most
one reverse success, and no unresolved paired outcomes. Do not substitute a
better cell after seeing results. This is a descriptive stopping decision,
not a significance test or impossibility theorem. If it fails, close the current
reference-support tuning cycle; regardless of outcome promote no default and
stop at this checkpoint.

## Audits, resources and reproduction

Preserve old source/checkouts/artifacts. Bind predecessor completion, source
bundle and six exact-head CI jobs; freeze all new package source/settings/runtime
before scientific fitting. Independently reconstruct raw laws/joins and group
exclusions, training preprocessing, class covariance/likelihoods, ranks/ties,
all actual residual derivatives/Adam updates and frozen predictions/decisions.
Numerical artifacts use no pickle. Exact replay comparisons include all retained
scientific arrays. Retain failures without replacement; incomplete inference is
withheld. Guard changes require an explicit recorded amendment.

Use at most two single-thread workers, six GiB of total new local scientific and
engineering evidence and eight GiB free. A measured full-settings preflight must
project main execution/audit/report below ninety serial minutes (25% allowance,
no parallel-speed credit); preflight has a twenty-minute serial ceiling. Count
staging twice, add a 25% scientific storage allowance and 128 MiB reserve.
Engineering regressions and CI are separately reported. Keep detailed evidence
local and publish code/docs/invented fixtures/figures/small aggregates only.

```bash
super-deli benchmark-known-reference bind \
  --archive-root /path/to/completed-observed-feasibility-study --output-dir known-study
```

Place the approved plan at `known-study/approved-plan.md`, then `freeze` with
`--study-config examples/known-reference/study.json`. Run/audit `--phase preflight`,
then `gate --staging-bytes MEASURED_BYTES`. Only a passing gate admits main
run/audit/report. All actions require the same archive/output roots. Completed
units are sealed, verified and skipped on continuation. Test installed behavior
outside the checkout using:

```bash
python examples/known-reference/smoke.py --output-dir invented-known-check
```

The smoke specimen data and its trained heads are explicitly invented engineering
fixtures, never scientific performance evidence. Source-bound jobs and completion
markers are retained locally. Verify Python 3.11/3.12 tests, package allowlists,
all six published-head CI jobs and before/after preservation, then stop.
