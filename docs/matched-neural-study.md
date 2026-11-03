# Matched morphology–ecology neural integration protocol

Status: approved by the user on 2026-11-02. The registered rule and scientific
settings must not be tuned after preflight or main outcomes.
Current master, verified 2026-11-02: cc3d7594959e5cb159a058fef46ea850e55880f2.

## 1. Product and input boundary

Add a separate opt-in `cluster-matched-neural` CLI and corresponding Python
fit/workflow/load interfaces for exactly two selected views: continuous specimen
morphology and individually measured continuous ecology. Both views must have
one observation per same biological individual, declared by explicit unit IDs.
Require the independent-individual assumption and an explicit individual-level
ecology declaration. Those declarations do not independently verify sampling.

Use the existing typed dataset/ingestion contracts. Align by unit ID, never CSV
row position or a reference label. Canonical fitting/output order is sorted unit
ID. Retain both original observation-ID lists, source row indices, selected
feature orders and the complete alignment ledger. Duplicate, repeated, missing,
extra or unmatched selected-view unit IDs cause a clear rejection before fitting;
there is no implicit intersection, aggregation, copying or fallback to one view.
Allow other modalities in a dataset to remain unselected and unread.

Support >=10 matched units, 3–32 explicitly selected continuous analytical traits
per view, and at most 64 total. Every selected cell must be finite and observed.
Reject counts, categories, covariate/reference columns and missing cells. Record
and exclude fit-constant features; require at least three retained traits in
each view, including in every subsample. Preserve all source values.

Population/locality ecology, repeated/unequal sampling, grouped or unmatched
units, genomic/shape neural inputs, missing data and empirical fitting are out
of scope. Existing workflows supporting those separate situations stay intact.
Do not contact authors, survey custodians or anyone else.

## 2. Fixed model and modular implementation

Use a separately versioned configuration/model family and isolate new alignment,
view preprocessing, frozen transforms, workflow output and study modules.
Reuse existing ingestion, CPU neural training, NumPy forward transforms,
HDBSCAN fitting and native subsampling accumulators. Keep existing numerical
trainers/configurations unchanged; require exact legacy parity for any necessary
shared-helper extraction. Keep PyTorch optional and help/config imports lazy.

For each view, estimate raw feature means and population SDs from fit units only.
After dropping constants, let p_M,p_E be retained feature counts and D=p_M+p_E.
Standardize each retained raw feature, then multiply its whole view by:

    w_M = sqrt(D / (2*p_M)); w_E = sqrt(D / (2*p_E)).

Concatenate morphology followed by ecology in fixed selected-feature order.
Apply these weights AFTER standardization. Reuse the trainer directly on this
weighted matrix, without a second input standardization that cancels weights.
Its existing mean-squared reconstruction objective equals the mean of the two
per-view standardized reconstruction MSEs, independent of raw column count.
This balances the loss; it does not promise equal learned biological influence
or invariance of fitted weights to duplicated or reordered features.

Reuse the unchanged ordinary autoencoder: D->16->8->2 encoder, 2->8->16->D
decoder, tanh hidden layers, linear bottleneck/output, float64 CPU full-batch
Adam, learning rate 0.001, weight decay 0.0001 and exactly 500 updates. No PCA
before the encoder, denoising, early stopping, tuning, winning-seed search or
training with reference/generating labels. Preserve initial and 500 post-update
losses; report weighted overall and separate per-view reconstruction losses.

Standardize the two latent coordinates using fit-unit means/SDs. Retain existing
collapse/rank/workspace guards and HDBSCAN settings: min_cluster_size=5,
min_samples=5 (including self under the existing implementation), EOM selection,
allow_single_cluster=true, alpha=1 and epsilon=0. K is inferred by this existing
density rule; -1 remains noise/unassigned, and valid all-noise K=0 is explicit.
Do not add Gaussian-reference spectral screening, count forcing or GMM tests.

Retain numeric network weights, both view transforms/constant exclusions,
weights, ordered identities, latent scaling, training diagnostics and provenance
without pickle. A saved model can encode and reconstruct complete paired new
rows using frozen parameters, including inverse view weights and restoring
training constants to their fitted means. It does not assign HDBSCAN species
labels or confidence to unseen individuals. Analysis groups are transductive.

Write assignments.csv, a source/identity manifest, summary.json, numeric model
and stability artifacts, plus one local HTML report with latent groups/noise,
per-view reconstruction, coverage and separate uncertainty meanings. No automatic
model winner or validated species partition is produced.

## 3. Matched native subsampling

The opt-in workflow supports the existing 80% whole-unit subsampling without
replacement. Select an individual's rows in BOTH views together. Within every
draw, refit per-view means/scales/constants/weights, network, latent scaling and
HDBSCAN. Do not use the point model's preprocessing in a refit. Keep neural
initialization fixed within a workflow; the separate subsampling seed controls
which individuals are included. Preserve source identities and failure evidence.

Report included-unit K frequencies, label-invariant coassignment with successful
pair-inclusion denominators, noise/coverage and failed-outcome bounds. Unseen
pairs are unavailable, not zero. Retain failed draws without replacements;
finite loss is not convergence proof and stability is not species probability.
No significance calibration or new p-values. Default n_subsamples remains zero;
all existing defaults, serialization and historical p-values remain unchanged.

## 4. Frozen, independent validation grid

Freeze generators, settings, seed manifest, source archive and numerical runtime
before generating any preflight/main data. Use fresh synthetic data only. Two
sample sizes, N=64/256, and ten independent datasets per law/size give 160 main
observations. There is one fixed measurement-noise setting, not a tuning grid.

All laws have four morphology and eight ecology traits, intentionally unequal
column counts. Use fixed row-normalized loadings:

    L4: (1,0), (0,1), (1,1)/sqrt(2), (1,-1)/sqrt(2).
    L8: L4 plus (1,2), (2,1), (1,-2), (-2,1), each divided by sqrt(5).
    X_M = U_M @ L4.T + epsilon_M; X_E = U_E @ L8.T + epsilon_E.

All epsilon cells are independent N(0,0.15^2). Except where specified, Z and
eta_M/eta_E are independent two-dimensional standard Gaussians. Four group
centers C_g are (-3,-3),(-3,3),(3,-3),(3,3), with equal fixed class counts.
Use eight laws:

1. Independent Gaussian control: U_M=eta_M, U_E=eta_E; one generating group.
2. Correlated Gaussian control: U_M=Z+0.5*eta_M, U_E=Z+0.5*eta_E; one group.
3. Redundant four groups: U_M=C_g+eta_M, U_E=C_g+eta_E; shared g truth.
4. Complementary four groups: equal combinations a,b in {-1,+1};
   U_M=(3*a,0)+eta_M, U_E=(0,3*b)+eta_E. Joint truth is (a,b); each view's
   own binary signal is separately documented and never a fitting input.
5. Morphology-only four groups: U_M=C_g+eta_M, U_E=eta_E independent of g.
   Shared generating truth is g; ecology alone deliberately has no g signal.
6. Discordant views: equally represented combinations g,h in {0,1,2,3};
   U_M=C_g+eta_M, U_E=C_h+eta_E. Retain separate view labels for descriptive
   agreement, without inventing one joint species truth or primary correct K.
7. Heavy-tail stress: replace Z by independent t3/sqrt(3) coordinates in law 2;
   retain Gaussian eta terms. No discrete truth or formal species-error score.
8. Moons stress: equal halves from (cos(t),sin(t)) and
   (1-cos(t),0.5-sin(t)), t~Uniform(0,pi), with independent latent N(0,0.08^2)
   perturbations. Set U_M=Z+0.1*eta_M, U_E=Z+0.1*eta_E. Retain two generating
   labels for secondary recovery diagnostics; do not tune to repair moons.

Shuffle rows, then present the ecology CSV in a separately shuffled order to
exercise ID joins without changing membership. Use unique invented IDs. Neural
inputs never contain truth, origin labels, case identifiers or row-order proxies.
Main seed roots: generation 96211, cell noise 96217, permutations 96223,
subsampling 96231. Preflight roots: 96241/96247/96253/96259 respectively. Role,
law index, N and replicate index extend each SeedSequence path. Record all paths
and all individual subsample IDs. Keep these streams separate from engineering
fixtures and past studies. Freeze all three neural seeds: 0,101,202.

Compare 14 point arms per observation:
- Paired balanced autoencoder/HDBSCAN, all three seeds (3).
- Existing single-view autoencoder/HDBSCAN for each view, all three seeds (6).
- Existing integrative PCA/GMM with both views (1).
- Existing single-view PCA/GMM for each view (2).
- Existing single-view PCA/HDBSCAN for each view (2).

Freeze existing baseline settings: two PCs per view and jointly for integrative
PCA/GMM, equal integrative view weights, full-covariance BIC selection K=1..4,
minimum cluster size five, ten starts, model seed 71323, and otherwise unchanged
settings. GMM workflows are point-only (bootstrap count zero). PCA/HDBSCAN uses
two PCs and unchanged density settings. Do not compare absolute BIC across
representations or infer a unique causal mechanism from whole-workflow changes.

## 5. Exact run budgets and hard gates

Main point workflows: 160*14 = 2,240. Attach five matched 80% subsampling draws
to the first observation in each N cell for laws 1,3,4,5: eight observations.
All nine neural and both PCA/HDBSCAN arms share those same individual draws,
for 8*5*11 = 440 additional complete refits. GMM stability stays uncomputed.
Replay all 14 point arms on the first independent-Gaussian N=64 and first
complementary-four-group N=256 observation: 28 exact point replays.
TOTAL MAIN: 2,708 workflow fits (1,818 neural and 890 classical). The 486 GMM
searches contain 1,944 K-specific fits; 2,222 HDBSCAN fits accompany the other
workflows. These component counts are distinct from whole-workflow counts.

Separate preflight: one observation per law/N, or 16*14=224 points. Use five
matched draws for all 11 supported arms on four preflight observations:
independent-Gaussian N=64, complementary N=64, redundant N=256, morphology-only
N=256, giving 220 refits. Add the same two-cell all-arm point replays: 28.
TOTAL PREFLIGHT: 472 workflow fits (342 neural and 130 classical). Use the same
full architecture, 500 updates, density rules and five-draw settings as main.

Independently audit the preflight before the main gate. Estimate main execution
PLUS complete numerical audit conservatively from worst measured timings by N,
arm and fit/refit role, counting reference-training work and a 25% allowance;
no assumed parallel speedup. At most two single-thread workers. Require projected
main plus audit under two hours, a measured 2 GiB cap on all newly retained local
evidence (including preflight, test archives, logs and verification staging),
and at least 8 GiB free throughout. Currently about 206 GiB is free; this does
not relax the evidence cap. Verify lossless archives before replacing only this
checkpoint's generated scratch copies. Preserve originals and earlier studies.

Stop after a failed hard gate; preserve the failure, sources and settings. A
budget/scope amendment requires user approval. Do not lower N, seeds, updates,
replicates, stability draws or audit coverage to obtain a passing gate.

## 6. Independent audit and scientific endpoints

Independently reconstruct all laws, source records, row permutations and joins;
check shared unit IDs and raw cells, per-view means/scales/constant exclusions,
loss weights and projections. Reconstruct neural forward/backward equations and
500 Adam updates in NumPy for every neural fit, retaining initial weights,
loss/gradient/update checks and final-parameter comparison. Reference arithmetic
is audit work, not additional candidate selection or new optimizer seeds. Reuse
existing numerical-reference primitives where suitable without calling the
training decision being checked. Independently check latent normalization,
density/noise outputs, native inclusion denominators, all candidate GMM
densities/criteria, exported rows and exact replays. Audit saved-model encode/
reconstruction against fitted values without adding fits.

Keep failures without replacements. Report correct K, full-unit ARI with noise
as unresolved singletons, assigned-only ARI, coverage, overpartitioning and
availability/failure bounds separately. Primary family: 56 Gaussian K>1 return
rates (2 laws*2 N*14 arms), plus 168 correct-K and joint correct-K/ARI>=0.8 rates
(3 four-group laws*2 N*14 arms*2 endpoints), or 224 rates. Give exact marginal and
Bonferroni simultaneous 95% intervals. Ten observations/cell provide bounded
pilot evidence, not universal performance certification. All three neural seeds
remain separate repeated measurements; never select a winning seed or treat
seeds or subsamples as independent biological datasets.

Freeze paired comparisons before generation: paired neural versus each
single-view neural at the same seed; paired neural versus integrative PCA/GMM;
all-request recovery/coverage and common-available agreement remain separate.
Use discordant-view labels descriptively and report heavy-tail/moons stress
separately. Five draws per selected dataset only exercise and describe the
stability workflow; they do not establish precise stability supports. A lower
reconstruction error is not evidence for better species recovery.

## 7. Compatibility, publication and stop

Use a new clean managed worktree from verified master. Preserve every existing
checkout, input, study, unresolved numerical failure and empirical evidence gate.
Run unit/integration and legacy-parity checks, dependency-light tests, scientific
and neural suites, package allowlist/source parity and outside-checkout saved-
model/CLI checks. Verify Python 3.11/3.12 with no optional-dependency skip replacing
required neural checks. Add strict ID/level/type/missingness negative tests and
row-permutation, constant-feature, unequal-feature-count and failure fixtures.

Publish only code/docs, invented fixtures, figures and small aggregates directly
to master; retain full evidence in a new local matched-neural checkpoint output
directory. Verify all final CI jobs on the exact master commit before reporting
completion. No automatic new default, new significance claim or empirical cohort
approval. Stop at this checkpoint regardless of whether integration improves
recovery. Broader modalities, grouped/unmatched integration, missing data and
new neural families remain later separately approved work.

## Approved numerical-audit amendment

After the original scientific runs, an independent serial NumPy optimizer
rollout missed the frozen final-weight guard on a heavy-tail stress case.
The failure and original source freezes are retained. No scientific fit or
candidate decision is replaced, and that serial guard is not widened.

The user separately approved the [per-step verification amendment](matched-neural-audit-amendment.md)
on 2026-11-02. It checks independent gradients and Adam recurrences on each
actual update and requires exact original initial/final weights and all 501
training losses. Twenty-four additional full-settings probes precede 2,160
training-only verification replays. These are numerical verification work,
not new statistical observations, model candidates or optimization seeds.
Both the original and amended resource gates, failures and source bindings
remain local. The scientific comparisons, thresholds and seed manifest above
are unchanged; the results report states this post-run amendment explicitly.
