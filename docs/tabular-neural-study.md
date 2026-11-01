# Frozen first-neural-model comparison

This is a bounded software and simulation validation of transductive density
groups. It does not establish species boundaries or presume that a neural
representation will outperform a classical one. Settings were approved before
preflight/main generation. Full source/data/model evidence remains local.

## Inputs and model arms

There are 80 fresh datasets: Gaussian control, four spherical groups, nonlinear
moons and a continuous cline, crossed with N=64/256 and ten independent replicates.
All tables have eight complete continuous features. Source two-dimensional laws
reuse `benchmarking.density_study.generate_raw` without changing parameters.

For Gaussian, four groups and cline, the eight-dimensional map is `z @ A.T`, with:

```text
A = [[1,0], [0,1], [1,1], [1,-1], [.5,1], [-1,.5], [2,.5], [.5,-2]]
```

Moons uses `[z1,z2,z1*z2,z1^2,z2^2,sin(z1),cos(z2),sin(z1+z2)]`.
Independent Normal(0, 0.15^2) measurement noise is added to each mapped coordinate.
The Gaussian control remains multivariate Gaussian; the cline has no discrete
reference partition or correct K. These numerical traits are constructed test
data rather than an empirical species benchmark.

Each observation receives five point fits: unchanged PCA/full-GMM, unchanged
PCA/HDBSCAN, and the fixed [autoencoder/HDBSCAN](tabular-neural.md) with neural
seeds 0, 101 and 202. All three neural seeds are retained. Seed zero is prespecified
for the primary comparison; no seed is chosen using recovery or reconstruction.
PCA retains two dimensions. GMM retains K=1..4, minimum size five, ten starts and
root model seed 71323. Density settings retain minimum cluster size/min_samples
five, EOM and allow_single_cluster true. Existing default behavior is preserved.

Data/noise/subsampling roots are 91003/91009/91019. Preflight namespace two and
main namespace three precede case index, sample size and replicate in each seed
sequence. Source order is shuffled by the unchanged generator. Reference labels
are stored separately, never in analytical configuration columns.

## Fixed fit accounting

| Stage | Count |
| --- | ---: |
| Main point workflows: 80 x 5 | 400 |
| First observation in eight cells, 20 paired 80% subsamples, five arms | 800 |
| Exact five-arm point replays for Gaussian N=64 and moons N=256 replicate zero | 10 |
| Main total | 1,210 |
| Separate preflight: eight new observations, point plus two subsamples, five arms | 120 |

A GMM workflow includes its whole K search. Subsampling is without replacement
and preserves original IDs. Every stage refits each workflow from raw selected
rows. The GMM study refits are not its ordinary unit-bootstrap estimator, and
do not alter its public workflow. Failed fits/draws remain in all attempt counts;
there are no replacement draws or post-outcome setting changes.

## Evaluation and audit

Datasets are the independent observations. Seeds and subsamples are repeated
measurements. For the three laws with partitions, report correct-K rates for
the two classical arms and neural seed zero: 18 primary rates. Give exact
pointwise and Bonferroni simultaneous 95% intervals, successful/requested
denominators, unresolved-outcome bounds and conservative interval envelopes.
Ten datasets per cell support only a small pilot validation.

Report K, full-unit and assigned-only ARI with coverage, noise, splits/merges,
Gaussian overpartitioning, cline partitions, runtime, training reconstruction,
neural initialization agreement and conditional inclusion-based subsampling.
Treat each noise observation as a distinct singleton for full-unit ARI. No
correct-K or ARI outcome is assigned to a cline. Reconstruction is training
error, not held-out predictive evidence. BIC is not compared with neural losses.

Independent audits reconstruct the generating laws, source/identity accounting,
normalization, PCA eigenvalues/scores, NumPy encoder/decoder forward values and
final training MSE, density invariants, selected-GMM densities/membership/BIC,
exact replay arrays, support inclusions/denominators and aggregate exports.
Gradient finite-difference tests and parent-code density parity are separate
development checks. Neither optimizer nor HDBSCAN is independently reimplemented.

## Resources and publication

At most two single-thread CPU workers, at least 8 GiB free, and at most 2 GiB
new retained scientific/build evidence including staging. A measured 120-fit
preflight must pass its numerical audit before the main study. Project storage
with a 25% allowance plus 128 MiB reserve, and project two-worker main runtime
with a 25% allowance against a two-hour limit. Retain a failed gate and request
an explicit amendment rather than silently shrinking the study.

Preserve prior checkouts/artifacts, defaults, schemas, historical p-values and
empirical evidence gates. Run no empirical/SPARUS analysis and contact nobody.
Publish only code/docs/invented fixtures/figures/small summaries after Python
3.11/3.12 tests, package checks and final master CI. Keep full simulations, fitted
weights and draws local. Stop at this checkpoint without promoting a new default.

## Reproduce the gated study

From a source checkout installed with the neural extra, choose a new output root
and copy this protocol there as `approved-plan.md` for a new reproduction record.
The executed run retains the original approved plan and its hash locally.

```bash
python -m super_deli.benchmarking.neural_study freeze --output-dir /absolute/new-study
python -m super_deli.benchmarking.neural_study preflight --output-dir /absolute/new-study
python -m super_deli.benchmarking.neural_study audit-preflight --output-dir /absolute/new-study
python -m super_deli.benchmarking.neural_study gate --output-dir /absolute/new-study --staging-bytes 0
python -m super_deli.benchmarking.neural_study main --output-dir /absolute/new-study
python -m super_deli.benchmarking.neural_study audit-main --output-dir /absolute/new-study
python -m super_deli.benchmarking.neural_study report --output-dir /absolute/new-study
```

Supply actual retained staging bytes when staging/build evidence exists. Exact
source/runtime hashes must agree with the freeze. Completed or failed output
directories are never reused; this checkpoint does not implement partial resume.

After the report phase, `examples/neural/report.py --study-dir /absolute/new-study
--repository /absolute/source-checkout` exports small aggregates and figures.
Install Matplotlib separately for this plotting script. The installed analysis
CLI's offline HTML report does not require Matplotlib.
