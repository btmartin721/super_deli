# Proposed checkpoint: Gaussian-reference spectral group-count feasibility

Status: approved by the user on 2026-11-01. Scientific outcomes must not alter
the frozen rule. This protocol authorizes the bounded feasibility checkpoint.

## Purpose and boundary

The completed 0.60.0.dev0 validation found frequent multiple-group returns for a
single rank-two Gaussian cloud, and zero correct two-group returns in its moons
cells. Freeze and evaluate one reference-adjusted spectral rule before extending
spectral stability or training more neural models. This is a benchmark-only,
versioned experimental method, not a replacement for the public model or a new
recommended species-delimitation workflow. A negative result is a completed
checkpoint; it will not trigger an unapproved search for a winning rule.

Baseline: master 13ec66dbf9bdf8e47ea087ff52c36f33b1fb728d, verified read-only.
Recheck master at implementation start. Preserve existing worktrees and all
prior inputs, artifacts, defaults, schemas, public model decisions and p-values.
Use a new clean managed worktree based on current master. Never contact study
authors or anyone else. No empirical analysis in this checkpoint.

## 1. Freeze the experimental rule

Support only the existing complete eight-feature synthetic tables at N=64/256.
The study concerns independent rows and an explicitly chosen graph; it does not
address repeated specimens, grouped sampling, missing data, genomes or shape.

For each table, fit a single Gaussian directly in raw feature space using the
sample mean and maximum-likelihood covariance (denominator N). Use a PSD-aware
spectral factorization with a documented machine-precision rank test:

    tolerance = 64 * machine_epsilon * max(N, p) * largest_eigenvalue.

Eigenvalues below negative tolerance cause withholding. Eigenvalues within
plus/minus tolerance are recorded as numerical zero, with the removed magnitude
and effective rank retained. Rank below two causes withholding. No positive
variance floor, shrinkage, dimension grid, fitted mixture or post-hoc repair is
allowed. This rank decision is numerical and is not a biological claim.

Draw raw Gaussian tables at the observed N from this fixed fitted distribution.
For every observed/reference table separately, refit raw-trait standardization
and reconstruct the exact existing five- or fifteen-other-neighbor graph,
normalized Laplacian and spectrum. Preserve identity-based ties, graph weights,
zero diagonal and the current numerical checks. Use both neighbor settings as
separate prespecified arms; never select one using outcomes. The same raw draw
is shared by both graph settings to make their comparison paired.

Let g_K = lambda_(K+1) - lambda_K for K=2,...,6, with ascending Laplacian
eigenvalues and one-based indexing. K=1 is handled by the reference screen,
not entered in the split-count maximum.

Use 49 independent pilot draws to estimate mean m_K and sample standard
deviation s_K of each gap. If any s_K <= 1e-12 or is nonfinite, withhold the
reference arm; do not floor or remove that coordinate. Freeze these pilot
normalizers. Define z_K = (g_K - m_K) / s_K and T = max_K z_K.

Use 99 further independent reference draws, disjoint from pilots, to obtain
T_b using those same frozen normalizers. Compute the experimental tail rank:

    p_reference = (1 + count(T_b >= T_observed)) / 100.

Use unrounded scores for this comparison. Retain the count, scores, ties and
Monte Carlo uncertainty. Any unresolved numerical fit/draw withholds the whole
reference result; no replacement draws or partially complete tail estimates.
This maximum accounts for the five searched gap coordinates for this fixed
graph only. It does not adjust selection between neighbor settings or methods.

If p_reference > 0.05, record 'no reference-supported split', represented by K=1
only as the experimental algorithm's point decision. Explicitly state that this
does not accept a Gaussian law, prove one population, or establish one species.
If p_reference <= 0.05, propose the unique K maximizing z_K, then apply existing
assignment/abstention guards: ties within 1e-8 withhold; K=6 withholds at the
search boundary; disconnected graphs require component count exactly K;
connected graphs use existing guarded spectral embedding/QR assignment; minimum
returned cluster size remains five. A failure of the proposed K never falls
back to another K or becomes K=1.

The new diagnostic tail rank is an approximate plug-in Gaussian-model departure
quantity. Estimated covariance, rank selection, finite pilot normalization and
finite reference simulation prevent any general finite-sample size guarantee.
It is not a probability that K is correct, a cluster support, a species test, or
an implementation of SigClust. No production significance workflow is changed.

## 2. Freeze fresh validation before generation

Reuse the unchanged, independently audited eight simulation laws from the
completed unknown-K study: rank-two Gaussian; full-eight-dimensional Gaussian;
four spherical groups; noisy moons; four unequal groups; heavy tails; continuous
cline; eight spherical groups outside the count-search range.

Generate ten fresh independent datasets for each law at N=64/256: 160 underlying
observations. Retain paired clean/noisy eight-trait maps, with unchanged noise
SD 0.15, for 320 analysis tables. Existing generator formulas, parameters and
identity shuffling stay fixed. All labels remain hidden from fitting/selection.

Use deterministic SeedSequence paths rooted at 95111 for main generation,
95117 for paired noise, 95131 for main reference draws, and disjoint roots
95143/95147/95153 for preflight generation/noise/reference draws. Record full
paths including law, sample size, replicate, condition and pilot/reference
role. Retain model seed 71323 for unchanged baselines. Invented unique IDs and
separate shuffle seeds are frozen in the manifest. No scientific fit precedes
this freeze; no rule changes after preflight or main outcomes.

Compare six point arms on each table:

1. Existing largest-gap spectral, five neighbors.
2. Existing largest-gap spectral, fifteen neighbors.
3. Reference-adjusted spectral, five neighbors.
4. Reference-adjusted spectral, fifteen neighbors.
5. Unchanged two-PC full-covariance BIC/GMM, K=1..4, ten starts, minimum size five.
6. Unchanged two-PC HDBSCAN with its existing noise/coverage policy.

Reuse each observed graph spectrum between its two selectors, but retain their
separate count and assignment decisions. Retain all baseline candidate evidence.
Count comparisons across raw traits and PCA also change representation; do not
attribute those differences only to the group-count rule.

## 3. Fixed execution budget and resource gates

Each table requires 296 reference graph-spectrum evaluations (148 draws times
two graphs), two observed graph spectra and two baseline workflows: 300 graph
or baseline evaluations. Main: 320 tables times 300 = 96,000 evaluations.
Add four exact complete-table replays, selected by frozen law/N/condition IDs,
including their reference draws: 1,200 evaluations. Total main budget: 97,200.
At most 1,296 observed spectral embedding/assignment steps accompany these
outputs (two selectors times two graphs times 324 tables); count these separately.
The point outputs number 1,920 before replays; outputs reuse spectra, not new
scientific inputs.

A separate preflight uses one observation per law and N, both conditions:
32 tables plus four exact complete-table replays = 10,800 evaluations, with at
most 144 observed spectral embedding/assignment steps. All full settings remain
unchanged. Audit the preflight before estimating storage/runtime. It is an
engineering feasibility test; do not tune the method from scientific outcomes.

At most two single-thread workers; measured new local evidence cap 2 GiB;
minimum 8 GiB free throughout; projected main execution plus independent audit
under one hour including a 25% allowance. Include preflight, verification logs,
packaged source snapshots and staging evidence in retained-byte accounting.
Stop at a failed hard gate and retain it. Any budget or scientific-scope change
requires an explicit amendment. Do not silently reduce draws/datasets or change
the method to pass the gate.

Preserve observed inputs, graphs, spectra, assignment evidence and all baseline
candidates. For every pilot/reference draw retain source-bound seed paths,
Gaussian parameters/factorization, raw-input and graph checksums, gap/spectrum
summaries and all decision statistics. Do not store every raw reference matrix
or graph redundantly; they must be exactly regenerable in the frozen numerical
environment and verified against checksums. Record this retention boundary.
Full evidence stays local in a new checkpoint output directory; publish no
per-specimen source data, fitted arrays or bulk artifacts.

## 4. Independent audits and interpretation

Independently reconstruct all generated observations and pilot/reference raw
draws; audit identities, covariance factorization, effective rank, standardization,
neighbor/tie rules, graph hashes and spectra. Independently check pilot means/SD,
maxima, exact tail accounting, proposal and abstention decisions, embedding
residuals, memberships, baseline densities/criteria and exact replays. Avoid
production decision helpers in reference decision arithmetic. Archive unresolved
failures without replacement. Numerical agreement does not establish biological
validity or a theorem for the reference rule.

Report all-request and available-only results separately, including failures,
no-supported-split returns, withheld count proposals and each baseline's noise
coverage. Report correct K and joint correct-K/ARI>=0.8 separately, plus full
ARI and availability bounds. Keep paired maps/methods as repeated measurements;
datasets, not reference draws, are the independent replication units.

Freeze 192 primary rates: 48 nontrivial-return rates for two Gaussian laws,
two N values, two conditions and six arms; plus 144 correct-K/joint-recovery
rates for three in-range discrete laws, two N, two conditions, six arms and
two endpoints. Report exact marginal and Bonferroni simultaneous 95% intervals,
paired differences, tail-rank Monte Carlo uncertainty, effective-rank diagnostics
and count/abstention distributions. Ten independent observations per cell make
this feasibility evidence, not definitive error-rate certification. Summarize
reference-screen pass rates separately from final returned partitions.

Heavy tails and clines have no discrete truth; report behavior without calling
it species false positives. Eight-group stress tests a deliberately truncated
search; no K expansion. Do not pool paired clean/noisy observations as independent.
Do not compare absolute likelihood/BIC across different representations.

Judge whether this frozen rule merits any later user-facing proposal; document
failures plainly. No automatic winner, configuration change, default promotion
or additional diagnostic/model grid is authorized. No new significance or
stability bootstraps outside the explicit reference draws; no neural training,
empirical fitting, dataset-cohort changes or contact with anyone.

## 5. Compatibility, publication and stop

Add modular benchmark-only tooling, invented unit fixtures, documentation and
small aggregate exports. Keep production spectral and other model source and
schema behavior unchanged, with regression/parity checks. Update the roadmap
with completed versus unresolved capabilities. Run appropriate unit/integration
suites, dependency-light checks, scientific checks, Python 3.11/3.12 checks,
package allowlist/source parity and installed CLI/API checks. Publish only
reviewed code/docs/figures/small summaries directly to master, under the user's
standing authorization, and verify the final master CI at the exact published
commit before reporting completion. Preserve existing checkouts/artifacts and
stop at this checkpoint regardless of scientific outcome.

## Primary methodological background

Liu et al. (2008), Statistical Significance of Clustering for High-Dimension,
Low-Sample Size Data, https://doi.org/10.1198/016214508000000454, and the authors'
SigClust documentation:
https://search.r-project.org/CRAN/refmans/sigclust/html/sigclust.html
These support the distinction between a specified single-Gaussian null and
biological species truth. They do not establish validity for the new graph
statistic, normalization, count rule or rank handling proposed here.

John et al. (2019/2020), Spectrum: fast density-aware spectral clustering for
single and multi-omic data, https://doi.org/10.1093/bioinformatics/btz704,
https://pmc.ncbi.nlm.nih.gov/articles/PMC7703791/ . This illustrates that spectral
count estimation is a separate methodological component; its algorithms and
reported performance are not being adopted or attributed to this checkpoint.
