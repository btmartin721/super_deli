# Fitted-null calibration with unequal grouped sampling

```bash
python -m pip install '.[clustering]'
super-deli calibrate-grouped-integrative /path/to/dataset.json \
  --calibration-config examples/grouped-calibration/calibration.json \
  --output-dir /path/to/new-calibration
```

Python 3.11/3.12 are supported. No SNPio or image dependency is required by
this continuous-data workflow. It complements the existing matched-individual,
known-reference and frozen-score tests without changing their hypotheses.

## Scientific hypothesis and input contract

Declare exactly two complete, finite, continuous analytical views:

- Morphology: one measurement vector per declared specimen, with an explicit
  `population_id` or `locality_id`. Individuals within a group may have unequal
  sample counts across groups. The view uses `aggregation="mean"`.
- Ecology: one vector measured directly per target population/locality, with
  `aggregation="none"`. Every declared group must have both views. Ecology is
  retained once per group through fitting and simulation.

Groups are independent sampling units. Counts are treated as fixed,
**non-informative** design: group effects and residual distributions do not
change with the number of sampled specimens. This is an explicit user assumption,
not something inferred or verified from the measurements. Individuals have iid
Gaussian morphology residuals, independent of group effects. There are no
individual cross-view pairs in this input contract.

For group g and morphology specimen j, the null is

`(U_g, E_g) ~ iid Normal(mu, B)` and `M_gj = U_g + epsilon_gj`,

where `epsilon_gj ~ iid Normal(0, R)`, B is a full between-group covariance
including morphology/ecology cross blocks, and R is a shared full within-group
morphology covariance. Both are unknown. With one ecology row per group,
ecological measurement error is not separately identified; the null models the
observed ecology vector as part of the joint between-group distribution.

Selected counts, categorical/ordinal fields, genomic/shape features, missing
cells/views, repeated specimen measurements, spatial/related groups, informative
sample counts, and heterogeneous within-group covariance are unsupported by this
null. Feature selection can exclude unused fields; generated declarations
contain only the selected continuous features and views. Source identifiers,
row orders, feature units, group links and counts are preserved. Group order
follows the first view, matching the integrative pipeline.

## Estimation before PCA

Let P morphology and Q ecology features define
`Z_g = (mean_j M_gj, E_g)` and
`V_g = B + blockdiag(R/n_g, 0_Q)`.
The residual scatter is
`W = sum_g sum_j (M_gj-mean M_g)(M_gj-mean M_g)'`.

The marginal negative log likelihood, up to parameter-independent constants, is

`0.5 * [sum_g {log|V_g| + (Z_g-mu)' inv(V_g) (Z_g-mu)}
         + (N-G) log|R| + trace(inv(R) W)]`.

This accounts for the precision of unequal group means rather than fitting an
iid Gaussian to their observed values. The mean is profiled by generalized
least squares. Internally, center/scale transformations improve conditioning;
parameters and data remain available in their original units. B and R use
Cholesky factors with log diagonals and analytic profile gradients. L-BFGS-B
uses three fixed, recorded starts. A moment correction supplies one start;
it is not the final covariance estimator. The best accepted numerical maximum
is selected; a global-optimum certificate is not claimed.

Acceptance requires G>P+Q, enough groups for two configured GMM components,
N-G>=P, nonsingular pooled residual and group-mean covariance, successful
optimization, standardized maximum gradient at most `gradient_tol`, and interior
between/within covariance eigenvalues above `rank_tol`. Diagonal log-factor
bounds [-20,20] guard numerical overflow; solutions touching them are rejected.
No covariance clipping, ridge, shrinkage or boundary repair is applied to an
accepted null. Singular/boundary or nonconverged fits produce `failure.json`,
zero clustering attempts and no p-value. Such failures are possible even with
valid Gaussian data, particularly for weak between-group variation.

The [mixed-model bootstrap literature](https://journal.r-project.org/articles/RJ-2023-015/)
describes generating fresh random effects and residuals from estimated Gaussian
laws. This implementation applies that principle to a global integrative
clustering statistic, with separate simulation validation.

## Full-pipeline bootstrap and inference

Fit the null once to raw observed measurements before PCA or clustering.
For each of B=99 independent bootstrap draws, generate fresh joint group
morphology/ecology effects and fresh morphology residuals, holding all observed
counts and identities fixed. Write raw CSV/declaration inputs and rerun ingestion,
feature selection/filtering, scaling, each source-view PCA, explicit morphology
score averaging by group, view normalization/weighting, fusion, joint PCA and
the complete eligible GMM K search. All group profiles receive equal weight in
the final GMM; the existing source-view PCA uses individual observations before
aggregation. Null parameters stay at their observed estimates: the statistic
does not require fitting another hierarchical null inside each draw.

The statistic is `T = BIC(K=1) - min eligible BIC(K=1..Kmax)`.
With b simulated statistics >= observed T, report `p=(b+1)/(B+1)` and reject when
`p <= alpha`, with default alpha=0.05. Ties count in the upper tail. Prespecify
feature selections, weights, dimensions, K range, GMM controls and seeds. The
observed and null point pipelines share one fixed analysis seed; raw draws use
separate recorded streams. No shared null bank, adaptive stopping, retries or
outcome-driven tuning is used. Stability bootstrap is disabled in this test
and remains separate evidence; ordinary integrative stability is still available.

P-values are **approximate** because mean and covariance components are fitted.
The plus-one convention and full refits do not provide the known-law exact-rank
guarantee. Binomial tail intervals address Monte Carlo uncertainty conditional
on fitted parameters and counts, not parameter estimation or misspecification.
There is no double bootstrap or general size guarantee. Every failed draw
withholds the primary p-value/decision; requested, attempted, successful,
failed and unattempted clustering counts remain explicit. Archive-limit failures
also preserve the actual attempt ledger. Outputs use new directories only;
source inputs are hashed before/after and never edited.

A small p-value indicates departure from this hierarchical Gaussian null.
Heavy tails, outliers, gradients and other misspecification can cause rejection.
It is not species-boundary significance, cluster-specific significance,
significance of the selected K, or a confidence level for biological species.
Non-rejection does not establish one species. No multiplicity correction across
separate analyses or feature/configuration searches is supplied.

## Prespecified validation and audit

```bash
# Repeat for correlated_gaussian, complementary and heavy_tailed.
super-deli benchmark-grouped-calibration examples/grouped-calibration/study.json \
  --case independent_gaussian --output-dir /path/to/study/independent_gaussian
super-deli analyze-grouped-calibration /path/to/study \
  --output-dir /path/to/new-audit
python scripts/plot_score_calibration.py /path/to/new-audit \
  --grouped-pipeline --output-dir /path/to/new-figures
```

The study fixes 25 independent datasets per case, 64 groups, four features/view,
two PCs/view and jointly, equal view weights, K=1..4, minimum size 5, 10 EM starts,
B=99 and analysis seed 71323. Master seed 20261024 derives independent observed
and per-dataset null streams. Thus 10,000 clustering fits are planned. Null-fit
failures leave the associated fits unattempted and remain in the ledger.

Counts are iid discrete uniform 1..4, drawn independently before Gaussian
variables and held fixed within each test. Between covariance for the independent
case is I8, within covariance is 0.75 I4, and the mean is zero. For the other
cases, between correlation is `0.6**abs(i-j)`, SDs range linearly 0.5..2 and means
-2..2; within covariance is `1.5 * 0.35**abs(i-j)` in raw morphology units.
The complementary alternative adds balanced, randomly assigned four-group shifts
of strength 3: morphology encodes one binary axis and ecology the other, each
with coefficients (1,0.7,0.5,0.3). The heavy-tail stress divides centered group
effects and morphology residual vectors by independent `sqrt(chi_square(3)/3)`
scales. Its scatter parameters are not its covariance. Neither group labels nor
truth files enter null estimation or clustering.

Gaussian cases estimate rejection rates under their declared law. Alternative
power concerns this strong constructed signal at the K ceiling, not membership
recovery or weak alternatives. Stress rejection is sensitivity to Gaussian
misspecification, not species/clustering type-I error. Report availability and
failure counts alongside rates. Wilson intervals and MCSE describe independent
available datasets; with missing tests these are conditional rates, not
unconditional error-control estimates. Bounds treating all withheld tests as
non-rejections versus rejections describe the remaining uncertainty and are not
confidence intervals.

The read-only auditor regenerates observed inputs and every null raw draw,
checks identities/counts and all archived bytes, and independently evaluates the
**dense individual-observation likelihood and covariance/mean scores** at fitted
parameters. It reconstructs fusion, joint projection, selected GMM
likelihood/BIC/membership, candidate selection, p-values and denominators.
It does not rerun the optimizer or clustering. View PCAs and nonselected GMM
likelihoods are archived/checksummed rather than independently refitted.
Outer failures retain ledger counts and error records without a full numerical
audit of a partial archive. This simulated validation does not establish
performance against empirical biological species ground truth.
