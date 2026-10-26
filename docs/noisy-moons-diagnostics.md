# Noisy-moons recovery diagnostic

This separate, versioned diagnostic investigates PCA/HDBSCAN recovery without
changing production models, defaults, sampling rules, historical p-values or
empirical evidence gates. Its settings are explanations to test, not a selected
replacement estimator. No empirical analysis or author contact is included.

## Questions and source boundary

The source is the completed [density checkpoint](tabular-density-results.md),
commit `85c5db3686d451673bb2a5a852ba86a6024e2e3d`. Bind its complete previously
audited 300-observation file inventory and configuration/software fingerprints.
Audit all 50 moons observations and 250 retained density subsamples without
refitting. Existing GMM point results remain context; no GMM study is rerun.

The source has two raw traits and two retained PCs. Verify source/identity/truth
correspondence, preprocessing and projected Euclidean distances, rather than
assume PCA truncation explains the failure. Reconstruct latent angles and noise
only by replaying the locked generator and matching exact analytical values.
Latent data and generating labels are kept in separate reference files.

Separate four mechanisms: scaling; finite-sample/noise-dependent density
connectivity; pruning of small branches; and cluster selection among surviving
branches. Truth-dependent graph/branch quantities are oracle diagnostics, not
selection rules available to a user without labels.

## Frozen arms and inputs

All arms retain two full-SVD PCs, Euclidean distance, alpha one, epsilon zero
and one thread. The unchanged baseline standardizes both traits and uses minimum
cluster size five, min_samples five (including self), EOM selection and permission
to select the root as a single cluster. Seven arms change one setting each:

| Arm | Sole change from baseline |
|---|---|
| no_single | allow_single_cluster=false |
| leaf | leaf selection instead of EOM |
| samples_2 | min_samples=2 |
| samples_10 | min_samples=10 |
| cluster_3 | min_cluster_size=3 |
| cluster_10 | min_cluster_size=10 |
| center_only | Center without variance scaling before the same two-PC PCA |

The centering arm changes the metric in raw units; it does not isolate PCA or
establish generally preferable units. HDBSCAN is deterministic conditional on
input/order, library implementation and settings. No EM optimization or
truth-initialized clustering enters this checkpoint. See the
[official API](https://scikit-learn.org/1.6/modules/generated/sklearn.cluster.HDBSCAN.html)
and [tree implementation](https://github.com/scikit-learn/scikit-learn/blob/1.6.1/sklearn/cluster/_hdbscan/_tree.pyx).

Apply all eight settings to the 50 archived moons inputs and 150 fresh independent
datasets: 25 each at N=64/256 under original-noise moons (SD 0.08), low-noise moons
(SD 0.02), and N(0,I) Gaussian controls. Each moons law uses balanced independent
uniform[0,pi] angles on (cos(t),sin(t)) and (1-cos(t),0.5-sin(t)), plus independent
isotropic Gaussian measurement noise. Different noise cells use independent
inputs, not paired latent innovations. All arms are paired within each dataset.

These 200 observations contribute 1,600 point pipelines; the 50 baseline archived
point replays are included in that count. Replay the first original subsample of
the first five archived observations at each N, adding ten fits: **1,610 main
pipelines**. These are exact replays, not new bootstrap draws. A separate preflight
uses one fresh observation per cell with eight arms and one archived draw per N:
**50 preflight fits**. Run no new calibration or resampling-stability study.

Freeze machine-readable settings and source hashes before any fresh generation.
Seed roots are 129721 (main), 129737 (preflight), and 129749 (data), with recorded
case/N/index coordinates and no replacement draws or outcome-driven changes.

## Hierarchy and geometry diagnostics

The adapter reuses existing numeric table validation/PCA and scikit-learn
HDBSCAN. It retains the hierarchy in that original fit, without a second estimator
fit for instrumentation. Explicit field/version guards cover the internal
linkage/condensed-tree format. Unsupported representations are unavailable;
input/numerical failures remain failures. Neither becomes K=1.

Core distance uses the self-inclusive min_samples convention. Compute dense
Euclidean and mutual-reachability graphs, an independent SciPy minimum spanning
tree, and between/within generating-group connectivity. Define the oracle
separation ratio as the minimum between-group edge divided by the largest
within-group MST bottleneck. A value above one permits a cut joining each true
group internally before their first bridge; it does not guarantee HDBSCAN will
select those groups. Gaussian controls have no two-group ratio.

Compute each condensed branch's stability as its retained mass integrated over
inverse-distance lifetime. Reconstruct the EOM ancestor-exclusive frontier and
compare a parent's own stability to its best child-frontier sum. Root eligibility
and leaf selection are separate policies. Audit reconstructed labels and density
strengths against the original fit. Normalize root-minus-child differences by
their sum for plots; do not compare absolute stability across raw/scaled metrics.

Using truth only after fitting, report the best Jaccard agreement for each
reference group among surviving branches and the best minimum Jaccard across a
pair of disjoint branches. These oracle maxima diagnose information remaining
in the hierarchy and never enter unsupervised selection. Retain branch traces,
linkage, condensation, scores and core/MST arrays locally. Full pairwise distance
matrices can be reconstructed from lossless retained scores.

A read-only auditor independently checks standardization, PCA geometry,
core/MST/linkage distances, leaf-set condensation, branch stability integrals,
selection inequalities, declared settings, source replays, IDs, latent provenance,
labels/noise, failure/unattempted accounting and the ledger. Library tree-to-label
replay provides another consistency check without fitting; it is not an
independent complete HDBSCAN implementation. Known synthetic trees test root/
child selection explicitly. Ambiguous tied-edge identities are not forced to
match when MSTs are equivalent.

## Outcomes and uncertainty

Keep K=0/1/2/>2, assigned/noise coverage, splits/merges and agreement distinct.
For ARI each noise observation is an unassigned singleton; assigned-only ARI is
reported with coverage. Gaussian one-group ARI degenerates to zero after any
split/noise partition, so group count and coverage must be read separately.
Gaussian fragmentation/noise are algorithmic outcomes, not species/type-I error.

Report 48 fresh-cell correct-K proportions (eight arms x three laws x two N)
with simultaneous exact 95% intervals using Bonferroni and explicit failed-outcome
bounds. Correct K is two for moons and one for Gaussian. Archived comparisons are
exploratory. Other paired means and empirical SEs are descriptive, not confidence
intervals; datasets, not original subsamples or hierarchy branches, are independent
benchmark units. No p-values or a winning-configuration/default rule are added.

## CLI, storage and package boundary

Install `super-deli[clustering]`; figure reproduction additionally needs Matplotlib.
`examples/noisy-moons/study.json` contains the frozen declaration. The CLI exposes
`audit-noisy-moons-archive`, `diagnose-noisy-moons`, `audit-noisy-moons` and
`preflight-noisy-moons`. All accept the study declaration and an explicit
`--archive-root`. Fits require a passed `--archive-audit`, the main phase requires
an audited `--gate`, and all outputs must be new paths. See `--help` for each
command's required arguments. The installed smoke creates an explicitly invented
small archive; it never requests private data or contacts study authors.

Detailed evidence stays local under the run's artifact root. Require at least
8 GiB free, no more than four single-thread study workers and a measured 256 MiB
new-retained-artifact cap. Pass the audited full-settings preflight with a 25%
storage allowance before the main run. Retain a failed gate; budget amendments
require separate approval. Only code/docs/invented fixtures/small summaries and
figures are public, within 16 MiB. Package payloads remain explicitly allowlisted.
Run Python 3.11/3.12 scientific/foundation tests, distribution and installed-CLI
checks, publish directly to master, verify the final master CI and stop.

See the [findings](noisy-moons-results.md). Production fixes, broader density
adapters, automatic model selection and empirical species validation remain
separate work.
