# Fresh unknown-K spectral validation protocol

Status: approved by the user before implementation and scientific fits.
Base: master 3aeaa33925d07804842c57c0901f5fe29e62dcc6, verified live.

## Purpose and boundary

Add one separately versioned, opt-in research workflow that proposes a group
count from complete continuous morphology OR ecology without receiving truth.
Use an explicit eigengap heuristic, allow K=1, and retain abstentions/failures.
Test fresh data before treating favorable archival results as confirmation.
The rule is a heuristic, not significance calibration or a species-boundary test.

Preserve every existing model/default, configuration schema, historical p-value,
empirical gate, checkout and artifact. Leave the supplied-K archival engine and
its frozen sources unchanged. No neural retraining, new bootstrap/subsampling,
significance testing, empirical fitting, author contact or automatic winner.
Grouped/integrative adapters, genomic/shape support and the unified comparison
schema are outside this first spectral-model checkpoint. Statistical stability
support remains a later checkpoint; the new output must say point-only research
results, with no statistical uncertainty support for individual partitions yet.

## Implementation contract

1. Work in a new managed worktree from verified master. Record preservation
   inventories before changes. Add independent configuration/model/workflow and
   lazy CLI modules, using existing tabular ingestion and provenance contracts.
   Reuse helpers within the new workflow and benchmark; retain historical
   scientific implementations as versioned protocols rather than altering them.
2. Require one unique explicitly independent biological unit per row, explicit
   analytical continuous features, finite complete values, at least two selected
   nonconstant columns and at least 32 units. Reject missing/count/categorical
   inputs and repeated units; do not infer aggregation or fill missing values.
3. Standardize selected raw traits (training means and population SD, ddof=0).
   No PCA or neural representation enters the spectral model. This is a
   transductive point workflow; no invented out-of-sample prediction API.
4. Require graph-neighbor choice explicitly in its new configuration: five or
   fifteen OTHER units. Existing defaults remain unchanged. Exclude self,
   order exact distance ties by source identity, and use (B+B.T)/2 binary-neighbor
   affinity with zero diagonal and the symmetric normalized Laplacian.
5. Freeze the benchmark search to K=1..6, requiring the first seven eigenvalues.
   With ascending eigenvalues, score K by lambda[K+1]-lambda[K]. Use the unique
   largest gap only; withhold if its gap or its lead over the runner-up is <=1e-8.
   These tolerances identify numerical choices, not statistical uncertainty.
   A winner at K=6 withholds the partition as a search-boundary result. Retain
   raw scores, proposed K and exact reasons even when selected_k is null.
6. If proposed K=1 and the graph is connected, return a marked constant
   partition. This is heuristic K=1 selection, not proof of one species or a
   statistical acceptance of a null. If a disconnected graph has exactly the
   proposed K components, return a marked component partition. Other disconnected
   cases withhold; do not reconnect, merge or allocate extra clusters.
7. For connected selected K=2..5, retain actual ARPACK spectral_embedding
   (seed 71323, tol=0, drop_first=False), guarded cluster_qr, sparse graph,
   spectra and QR trace. Reuse the prior residual/Gram <=1e-8, projector <=1e-6,
   margin >1e-10*max(1,row maximum) and exact-label-count guards. A proposed
   partition with any group below five distinct units is withheld. Do not select
   the second-best K after any failed guard; retain numerical failures separately.
8. Export ordinary assignments only for available partitions, plus full status,
   scaling/identity provenance, gap trace, component counts and numerical checks.
   Distinguish withheld labels from HDBSCAN noise. No membership probability,
   confidence support, new p-value or calibrated error-control claim is added.

The largest-gap rule is a specific frozen experimental interpretation of the
standard eigengap heuristic, not a universally justified K estimator. The primary
literature warns that overlap/noise and graph construction complicate selection:
https://people.eecs.berkeley.edu/~jordan/sail/readings/Luxburg07_tutorial.pdf
(section 8.3). QR/library reference:
https://scikit-learn.org/1.6/modules/generated/sklearn.cluster.SpectralClustering.html

## Frozen fresh validation

Eight laws, N=64/256, twenty independent observations per cell: 320 observations.
Every observation has paired clean/noisy feature tables (640 tables).

1. Two-dimensional single Gaussian, mapped linearly to eight traits using the
   unchanged denoising-study linear map (rank-two Gaussian control).
2. Eight independent standard-normal raw traits (full-rank Gaussian control).
3. Four spherical groups using the unchanged existing generator and linear map.
4. Two noisy moons using the unchanged generator and nonlinear eight-trait map.
5. Four unequal-size/unequal-covariance groups using the existing generator/map.
6. One continuous heavy-tailed law: independent standardized t(3) latent axes,
   mapped by the same linear map. Report partitioning as stress, not species error.
7. The existing continuous-cline generator mapped linearly; retain no discrete
   generating partition and compute no fictional correct-K/ARI truth.
8. Eight equal-size spherical groups on a radius-eight regular octagon, SD=0.6,
   mapped linearly. Explicit search-limit stress: truth exceeds spectral Kmax=6
   and the unchanged GMM Kmax=4; HDBSCAN remains unconstrained by those caps.

For each underlying observation add paired independent N(0,0.15^2) raw-trait
measurement error to form the noisy table, preserving the clean table and mask
of added noise. The eight-dimensional Gaussian law uses direct raw draws.
For t(3), divide each latent axis by sqrt(3); no estimated normalization.
Shuffle rows with a separate deterministic permutation; preserve unique IDs.

Freeze fresh source seed root 93411, noise root 93427, main namespace 19 and
preflight namespace 17 before generating any scientific data. Key streams by
law index, N and replicate, using SeedSequence. No archived dataset is reused
for the fresh performance estimates. Existing generator implementations remain
unchanged and their version/source hashes are retained.

Four paired arms per table:

- Experimental spectral/eigengap with five neighbors.
- The same new rule with fifteen neighbors.
- Unchanged tabular PCA/full-covariance GMM: two PCs, BIC, K=1..4, minimum
  size five, ten starts, 500 EM iterations, tol=.001, reg=1e-6, seed 71323,
  standard scaling and n_bootstrap=0.
- Unchanged PCA/HDBSCAN: two PCs, minimum size/min_samples five (self included
  in min_samples), EOM, allow_single_cluster=True and n_subsamples=0.

No graph setting is selected automatically from agreement, labels or outcomes.
Both graph arms are experimental settings, not promoted defaults. Record
differences in representation, search range, small-cluster rules and noise policy.

Budget: 640 tables x four arms = 2,560 main point workflows. Add 32 exact
workflow replays: both conditions/all four arms for replicate zero of rank-two
Gaussian N64, full-rank Gaussian N256, four-spherical N64 and moons N256.
Total: 2,592 main workflow evaluations. No replacement or tuned second run.

## Separate preflight and hard gates

Generate one independent preflight observation per law/N cell: 16 observations,
32 tables and 128 workflows. Replay both conditions/all arms for rank-two
Gaussian N64 and moons N256: sixteen more workflows. Add ten separately marked
engineering timing probes for N64/256 x supplied K=2..6 on prespecified complete
Gaussian fixture coordinates; these probe worst-case embedding cost, never K
selection or scientific performance. Total preflight: 154 evaluations.
The probes use independent eight-trait normal fixtures keyed by [93431,N]; no
fixture searching or replacement is allowed to satisfy the gate.

Independently audit every preflight output and exact replay before the main run.
Timing must cover completed graph construction/actual embeddings at both N
values and every probe K, plus completed GMM and HDBSCAN workflows at both N.
Use the worst measured per-N costs and audit costs; conservatively assume every
spectral request requires the largest measured embedding and no saving from
abstention. Include independent audits and a 25% allowance; projected main run
must remain below one hour. At most two single-thread workers.

Require at least 8 GiB free, a measured 2 GiB new local-evidence cap including
staging and verified retained copies, a 25% size allowance and 128 MiB reserve.
Account for installed environments/caches separately. Preserve failed gates;
ask for an explicit amendment before changing scope/budgets after a failed gate.
Do not silently truncate the study or drop evidence to pass a gate.

## Evaluation and independent audit

Evaluate truth only after fitting. Report every requested/available/withheld/
failed denominator, proposed versus returned K, gap spectra, component-only
counts, size imbalance, splits/merges, HDBSCAN noise/coverage, and full-unit and
assigned-only ARI where truth exists. Retain ARI bounds [-1,1] for unavailable
partitions and explicitly counted common-available paired comparisons.

Freeze 128 primary rates: 32 Gaussian-control nontrivial-return rates (two laws
x two N x two conditions x four arms), and 96 alternative recovery rates
(three K=2/4 laws x two N x two conditions x four arms x two endpoints:
returned correct K, and returned correct K with ARI>=0.8). Abstentions/failures
are not correct returned partitions. For Gaussian nontrivial-return rates show
failure-aware bounds; distinguish returned K=1, abstention and numerical failure.
Use exact marginal and Bonferroni-adjusted 95% intervals across the 128 rates.
These measure the frozen algorithms under these laws, not universal false-positive
control. For stress laws report descriptive behavior separately; no species or
type-I-error interpretation of stress partitions. Twenty datasets/cell and
paired conditions/settings do not establish general biological validity.

Audit source/identity/noise joins, preprocessing, neighbor ties/affinities,
independent Laplacian spectra, all K scores and decision/abstention branches,
actual embedding residuals/projectors, QR/SVD labels, counts, metrics, intervals,
candidate GMM likelihood/BIC, HDBSCAN noise and every exact replay. Use separate
arithmetic/audit code and invented negative fixtures. Do not claim a second
complete implementation of the library algorithms.

Endpoint: publish the fresh results whether favorable or unfavorable, with at
most one targeted follow-up proposal. Do not promote a winning graph, change
thresholds/defaults, repair disconnected graphs or start another study here.

## Verification, publication and stop

Test new parsing/serialization, missing/invalid inputs, K=1, component paths,
search boundaries, numeric/tie/small-cluster abstentions, failure retention,
all outcome-independent decisions, resource gates, corruption detection,
replay and installed CLI behavior. Run the appropriate full regression suites,
dependency-light checks, lint/format and package allowlist/installed-wheel checks.
Verify Python 3.11/3.12 CI, including the new workflow and existing models.

Keep detailed raw data, coordinates, partitions, graph arrays, candidate fits,
audit evidence and failed cases local. Publish only code/docs, invented
fixtures, figures and small aggregates. Update the README/roadmap, push directly
to master with an expected-head check, verify final master CI and preserved
checkouts/artifacts, write a local completion record and stop at this checkpoint.
