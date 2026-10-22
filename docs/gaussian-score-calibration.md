# Conditional Gaussian calibration of integrative scores

```bash
python -m pip install '.[clustering]'
super-deli calibrate-integrative /path/to/completed-integrative-run \
  --calibration-config examples/calibration/calibration.json \
  --output-dir /path/to/new-calibration-output
```

Python 3.11/3.12 is supported. Calibration reads a completed
`cluster-integrative` run without changing its scores, assignments, stability,
or species-boundary evidence. A previously ingested genomic run can be tested
with `[clustering]` alone: SNPio and image codecs are not needed to read the
saved numerical representation. Output must be a new directory outside the
source run. The explicit `assume_iid_gaussian_scores=true` is a scientific
assumption, not an inference made by the software.

## Hypothesis and statistical boundary

The global null is that retained target profiles have independent, identically
distributed rows from one full multivariate Gaussian **in the frozen joint
score representation**. The alternative is a larger admissible full-covariance
Gaussian mixture within the prespecified K search. Population/locality profiles
require the same iid assumption. Unequal source sample counts can create unequal
precision; declaring group links does not make that assumption valid.

This is an approximate, fitted-null parametric-bootstrap diagnostic. It holds
raw-data filtering, missingness, source-member aggregation, view weights, GPA,
view PCA, joint PCA, retained dimensions, and cohort selection fixed. It does
not account for their estimation or selection uncertainty. Scores learned on
the same data need not satisfy the iid Gaussian null. Simulation checks directly
in score space do not establish full-pipeline calibration for genomic, shape,
or tabular inputs. No evolutionary or biological species null is supplied.

Rejecting a Gaussian can reflect heavy tails, skewness, outliers, continuous
heterogeneity or other model misspecification. It neither establishes species
boundaries nor validates individual clusters or a selected K. Failure to reject
does not establish one species or no biological structure. Bootstrap stability,
conditional membership, and this global p-value remain distinct evidence types.

Null-model choice is central to clustering significance, as discussed by
[Hennig and Lin](https://arxiv.org/abs/1502.02574). This implementation uses a
Gaussian GMM/BIC diagnostic; it does not implement their full framework or
[SigClust](https://pmc.ncbi.nlm.nih.gov/articles/PMC4706235/).

## Statistic and simulations

For a fixed score matrix X with N rows and d columns, the statistic is

`T = BIC(K=1) - min{BIC(K): K=1,...,Kmax, candidate eligible}`.

Thus T is nonnegative and is zero when K=1 wins. Every point and simulated
search includes all configured K candidates, using the existing convergence,
regularization, restart, and minimum hard-component-size rules. Inadmissible
components cannot contribute a lower BIC. A numerical failure of a feasible
candidate fails that search. Searching K is part of the tested statistic;
there is no extra test for each K or post hoc cluster.

The null uses the sample mean and full maximum-likelihood covariance with
divisor N. Rank-deficient/nonfinite inputs fail; covariance eigenvalues are not
silently clipped or replaced. Each of B iid simulated N-by-d Gaussian matrices
has its own recorded data and EM seeds. Gaussian parameters are fitted once to
the observed scores; each simulated matrix receives a fresh complete GMM search.
The original point seed is reused to reproduce the saved point search. A mismatch
in selected K or BIC improvement fails instead of attaching a test to a different
point result.

With b simulated statistics at least as large as the observed statistic,
`p = (b+1)/(B+1)`. Ties, including the atom at zero, enter the upper tail. The
default B=199 gives minimum p=0.005; zero p-values are impossible. Outputs record
whether the requested resolution can reach alpha. The plus-one convention follows
[Phipson and Smyth](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf).
**It does not make a plug-in, estimated-null test exact.** Finite-sample size is
checked empirically under the declared generators, not guaranteed generally.

Alpha defaults to 0.05 and must be prespecified. The test is global within one
configured representation. Its explicit multiplicity treatment is `none` across
different analyses: choosing among view subsets, dimensions, weights or datasets
after inspecting p-values is not covered. Independent validation does not tune
weights, K limits, thresholds or generators in response to results.

## Numerical uncertainty and failures

The 95% Clopper-Pearson binomial interval describes the null tail probability
conditional on the observed statistic and estimated null parameters. It
quantifies finite-simulation error only; it is not a confidence interval for
species support or a correction for null-parameter or PCA uncertainty.

Every requested null draw is retained without retry. If f draws fail, the main
p-value, reject decision, and conditional tail interval are withheld. The
plus-one failed-outcome bounds `[(b+1)/(B+1),(b+f+1)/(B+1)]` identify unresolved
simulation outcomes and are not confidence intervals. The CLI exits 1 for this
incomplete inference; invalid inputs/source mismatches exit 2. A completed study
may contain incomplete or failed dataset attempts, with their denominators retained.

## Files and API

The adapter checks configuration and numerical SHA-256 values, reconstructs
scores from saved fusion/PCA parameters, verifies score CSV identities/order,
and reproduces the point search. Hashes of the read source files are checked
again before export. Independent output contains:

- `summary.json`: assumptions, statistic, approximate p-value, resolution,
  conditional Monte Carlo interval, failure accounting, scoped interpretation,
  source hashes, policies and dependency versions.
- `null_model.npz`: observed scores, fitted mean and covariance; numeric arrays
  only, loaded with `allow_pickle=False`.
- `null_replicates.json`: every requested draw's seeds, array checksum, candidates,
  statistic and selected K, or its failure reason.
- `evidence.json`: the existing `Evidence` contract with target
  `single_gaussian_frozen_scores`, explicit null and multiplicity. No original
  `species_boundaries` evidence is changed.

```python
from super_deli.models.calibration_config import GaussianCalibrationConfig
from super_deli.workflows.score_calibration import run_integrative_calibration

summary = run_integrative_calibration(
    "/path/to/completed-integrative-run",
    config=GaussianCalibrationConfig(
        assume_iid_gaussian_scores=True,
        n_simulations=199,
        alpha=0.05,
        random_seed=20261021,
    ),
    output_dir="/path/to/new-calibration-output",
)
```

The numerical API `models.gaussian_calibration.calibrate_gaussian_scores` accepts
complete numeric scores plus an existing validated clustering configuration and
the separate calibration policy. It performs no raw-data ingestion or PCA.
The original configuration's GMM controls, dimensions and thread limit are used;
its `n_bootstrap` and assignment threshold do not enter the global statistic.
Workspace/log-size guards reject oversized requests. Stored records require
O(B*Kmax) space; numerical simulation is streamed one matrix at a time.

## Prespecified validation study

The [study declaration](../examples/calibration/study.json) uses 100 independent
datasets for each of four scenarios, N=64, d=2, B=199, K=1..4, minimum component
size 5, 10 EM starts, regularization 1e-6, and alpha=0.05. Data and calibration
streams derive separately from master seed 20261021 and stable scenario/replicate
keys. The nested calibration seed is replaced by each derived per-dataset seed.
These seeds are independent of the earlier integrative campaign.

| Scenario | Generator | Interpretation of rejection rate |
| --- | --- | --- |
| Spherical Gaussian | N(0,I) | Type-I error under this null |
| Correlated Gaussian | Mean (2,-1), covariance [[4,1.6],[1.6,1]] | Type-I error under this null |
| Separated mixture | iid equal-weight membership at four (+/-4,+/-4) means, within-group SD 0.35 | Power against this specified alternative |
| Heavy-tailed stress | Unimodal elliptical multivariate Student t with 3 df and the correlated scatter matrix | Gaussian misspecification sensitivity, not species false positives |

The study directly generates score matrices. No upstream dimensionality selection,
raw-modality simulations, grouped sampling or spatial dependence is assessed.
The K ceiling equals the mixture component count; recovery beyond that ceiling
and robustness to other dimensions/sample sizes remain untested.

Run each scenario into its own new directory beneath a shared study root:

```bash
super-deli benchmark-score-calibration examples/calibration/study.json \
  --scenario spherical_gaussian --output-dir /path/to/study/spherical_gaussian
```

Repeat for `correlated_gaussian`, `separated_mixture`, and `heavy_tailed`.
Each scenario may run as a separate single-threaded process. The independent
auditor requires all four complete shards with the same declaration:

```bash
super-deli analyze-score-calibration /path/to/study \
  --output-dir /path/to/new-audit
```

It reconstructs generated observed scores, fitted null parameters, all random
streams and draw checksums, candidate selection, statistics, p-values and dataset
event rates without refitting. It checks ledgers and raw artifact hashes, then
exports dataset/rate CSVs and a full SHA-256 inventory. Means and MCSE weight
independent datasets equally. Pointwise, unadjusted Wilson 95% intervals quantify
Monte Carlo uncertainty of rejection rates, not simultaneous validation claims.
Incomplete p-values have explicit availability denominators. Empirical rates
with 100 datasets are imprecise and cannot certify universal error control.

The auditor checks selection bookkeeping and statistics from saved candidate BICs;
it does not re-estimate candidate likelihoods. Incomplete tests make event rates
conditional on availability, which can bias error-rate comparisons.

See the [completed validation results](gaussian-score-calibration-results.md) for
audited rates, figures, provenance and interpretation limits.
