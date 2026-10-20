# Empirical fitted-null calibration of continuous integrative data

```bash
python -m pip install '.[clustering]'
super-deli calibrate-empirical-integrative /path/to/dataset.json \
  --calibration-config examples/empirical-calibration/calibration.json \
  --output-dir /path/to/new-calibration
```

This checkpoint supports **matched individuals with complete continuous
morphology and ecology**. It fits a joint Gaussian to their selected raw
measurements before any PCA. The fitted model preserves estimated cross-view
covariance. Each bootstrap dataset is written as new raw CSV inputs and receives
the complete integrative point pipeline. Python 3.11/3.12 are supported; SNPio
and image dependencies are not needed by this command.

## Scientific boundary

The global null is that independent specimen rows, concatenated across the two
selected raw views, follow one multivariate Gaussian with unknown mean and full
covariance. `assume_iid_gaussian_raw_views=true` explicitly asserts this model.
It is a scientific assumption, not a conclusion drawn by the software.

Parameters are estimated from the observed measurements. The resulting
parametric-bootstrap p-value is **approximate**. The plus-one calculation and
full pipeline refits do not restore the known-law exact-rank argument. Finite
simulation intervals do not quantify fitted-parameter uncertainty or model
misspecification. There is no double bootstrap or general size guarantee.

This tests departure from a fitted Gaussian population. Heavy tails, gradients,
outliers, skewness and other non-Gaussian structure can cause rejection without
biological species boundaries. Non-rejection does not establish one species.
It supplies neither cluster-specific significance nor significance of the
selected K. Bootstrap stability and conditional GMM membership remain separate
evidence types. Fitted-null cluster validation and the importance of choosing
which structure the null preserves are discussed by
[Hennig and Lin](https://arxiv.org/abs/1502.02574). This is a bounded Gaussian
baseline, not their entire flexible null-model framework.

The previous `calibrate-integrative` command conditions on frozen joint scores;
the previous `calibrate-reference-pipeline` uses a fully specified four-view
raw-data simulation law. Both remain available with their existing hypotheses.

## Input contract and fitted parameters

The integrative policy must select exactly one morphology and one ecology view,
both directly measured at specimen level. Every declared specimen must occur
exactly once in each selected table; matching uses explicit case-sensitive IDs,
not row position. No aggregation or ecological down-broadcast is performed.

Selected features must be continuous, finite and complete. Explicit feature
selection may omit analytical fields not intended for this test; no selected
count, categorical, genomic or landmark field is silently transformed into a
continuous Gaussian variable. Unselected fields and modalities are not modeled.
Generated null declarations contain only the selected features and views.
Original observation IDs, specimen IDs, feature names, units and source row
orders are preserved in the null layout.

Genomic/count/shape distributions, unequal group sampling, repeated measures,
missing cells, missing views, related targets and spatial dependence require
separate empirical-null extensions. This command rejects unsupported selected
inputs. Other Super Deli workflows retain their existing broader ingestion and
group-aggregation support.

For an N-by-P raw concatenated matrix X, estimate
`mu = mean(X, axis=0)` and `Sigma = (X-mu).T @ (X-mu) / N`.
The full covariance includes morphology/ecology cross blocks. N must exceed P
and provide at least two feasible minimum-size GMM components. Dense workspace
is budgeted. Constant features, nonfinite results and numerically singular
covariance are rejected; covariance eigenvalues are not clipped and no shrinkage
or ridge is added to the fitted null.

Rank assessment uses the fitted correlation matrix, avoiding dependence on
measurement units. Sampling factors that correlation and rescales it by the
fitted marginal standard deviations. Raw units and estimated full covariance
are retained. This null is fitted once per observed dataset, before its point
clustering; it is not fitted to clusters, labels, PCs or stability results.

## Bootstrap and inference

Each draw has N independent rows from the fitted joint Gaussian and is split
back into raw views using the frozen layout. This is not ordinary resampling
of observed rows, nor a permutation of view identities. The workflow reruns
CSV ingestion, analytical feature selection, preprocessing/feature filtering,
scaling, both view PCAs, block normalization/fusion, joint PCA and every configured
GMM K candidate. Null parameters are held at their observed estimates for these
draws. Independent observed datasets in validation receive independently fitted
nulls. The statistic does not require fitting another null within each draw.

Dimensions, feature selections, weights, GMM controls, alpha and the analysis
seed are fixed in advance. The observed and null point fits use the identical
model seed; raw simulation seeds have separate recorded streams. There is no
outcome-driven tuning, shared null bank, adaptive stopping or retry. Null draws
are streamed through temporary staging and archived after fitting.

The global statistic is
`T = BIC(K=1) - min eligible BIC(K=1..Kmax)`.
K=1 is included, so T is nonnegative and can have a large atom at zero. Existing
convergence, regularization, restarts and minimum hard-component-size rules
apply. A feasible candidate's numerical failure fails its complete search.
Regularization of GMM candidates is separate from the unregularized raw null.

With b of B draw statistics at least as large as observed T,
`p=(b+1)/(B+1)`. Ties enter the upper tail. Default B=99 gives minimum p=0.01;
the decision is `p <= alpha` with default alpha=0.05. See
[Phipson and Smyth](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf) for the
plus-one convention. Here fitted nuisance parameters make the test approximate.
There is no multiplicity correction across datasets or alternative analysis
settings; the complete K search belongs to one prespecified global statistic.

The Clopper-Pearson 95% interval describes the fitted-null tail probability
conditional on observed T and estimated parameters. It is a Monte Carlo interval,
not species confidence. Any failed requested null draw withholds the primary
p-value, reject decision and interval. The failed-outcome identification bounds
are `[(b+1)/(B+1),(b+f+1)/(B+1)]`; they are not confidence intervals. A failed
observed point search leaves all B draws unattempted. Invalid raw-null fits fail
before point clustering. Attempt counts remain explicit in either case.

Stability must be disabled (`n_bootstrap=0`) for calibration. A separate ordinary
integrative run may provide stability using the same point-analysis settings.

## Validation study

The [declaration](../examples/empirical-calibration/study.json) fixes N=64,
four continuous features per view, two PCs per view and jointly, equal weights,
K=1..4, minimum component size 5, 10 EM starts, B=99 and alpha=0.05. Master seed
20261023 supplies independent observed-data and per-dataset bootstrap streams;
the point analysis seed is 71323. There are 25 observed datasets in each case:

| Case | Raw generator | Rejection-rate meaning |
| --- | --- | --- |
| Spherical Gaussian | Independent standard normal coordinates in eight dimensions | Type-I error under this null |
| Correlated Gaussian | AR(1) correlation 0.6, marginal SDs evenly spaced 0.5..2 and means -2..2 | Type-I error under this null |
| Complementary | Balanced four groups; each view encodes one binary group axis, shifts 3 times (1,0.7,0.5,0.3), independent unit normal noise | Power for this specified alternative |
| Heavy-tailed | Unimodal elliptical multivariate t3 with the correlated Gaussian scatter and means | Gaussian misspecification sensitivity, not species/clustering type-I error |

Every observed dataset fits its own null and receives 99 fresh draws: 100
observed and 9,900 bootstrap clustering fits are planned. This is simulated
validation of an estimator fitted to observed data; no empirical biological
species ground truth or empirical biological performance claim is supplied.
The K ceiling equals the constructed alternative group count.

```bash
super-deli benchmark-empirical-calibration examples/empirical-calibration/study.json \
  --case spherical_gaussian --output-dir /path/to/study/spherical_gaussian
# Repeat for correlated_gaussian, complementary and heavy_tailed.
super-deli analyze-empirical-calibration /path/to/study \
  --output-dir /path/to/new-analysis
python scripts/plot_score_calibration.py /path/to/new-analysis \
  --empirical-pipeline --output-dir /path/to/new-figures
```

Four case processes can each use one numerical thread. The small smoke
declaration is only for installation checks. Failed attempts are not replaced.
Dataset-weighted rejection rates include explicit available/missing denominators,
pointwise unadjusted Wilson 95% intervals and Monte Carlo standard errors.
Availability can be selective; rates among successful tests alone can be biased.
With 25 datasets per case these intervals are broad and do not certify universal
nominal size or robustness to other sample sizes, dimensions or laws.

## Artifacts and audit

Each calibration writes configuration, initial null ingestion, a numerical
`null_model.npz` containing raw values/mean/covariance (no pickle), `layout.json`,
point and per-draw records, inference/evidence, and an inventoried
`pipeline_artifacts.zip`. The ZIP preserves raw null CSV inputs, copied observed
inputs, every pipeline's preprocessing/PCA/profile/fusion/GMM parameters and K
logs. `max_archive_bytes` limits uncompressed ZIP content per test. Initial null
ingestion and parameter files are separate; total study storage grows with N,
P, B and the number of datasets. Existing output directories are refused.

The auditor regenerates observed datasets and all raw null draws, independently
reconstructs raw MLE parameters and p-values, verifies archive/parameter bytes,
and checks raw source arrays, identities, block fusion, joint projection,
selected GMM likelihood/BIC/membership and K-selection bookkeeping without
clustering refits. View PCA and nonselected likelihoods are preserved and
checksummed, not independently refitted. Outer failures preserve their attempt
ledgers without a complete numerical re-audit of partial archives. Original
source hashes are verified and audit output is written outside study shards.

See the [completed validation results](empirical-null-calibration-results.md).
