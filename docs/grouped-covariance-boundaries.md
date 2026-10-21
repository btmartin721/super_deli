# Covariance boundaries in grouped fitted-null calibration

```bash
python -m pip install '.[clustering]'
super-deli calibrate-grouped-integrative /path/to/dataset.json \
  --calibration-config examples/boundary-calibration/calibration.json \
  --output-dir /path/to/new-run
```

Python 3.11/3.12 remain supported. This checkpoint keeps the grouped continuous
Gaussian hierarchy and adds its natural positive-semidefinite covariance
closure. Schema 2 explicitly selects the new estimator and sampler; schema 1
retains the interior-only estimator, serialization and sampler used by #45.
Historical raw artifacts remain auditable. Existing input, fixed-count and
independent-group assumptions are unchanged; see the
[grouped input contract](grouped-null-calibration.md).

## Conditional formulation

Let ecology `E_g ~ N(mu_E,C)`. Conditional latent morphology follows
`U_g = mu_M + A(E_g-mu_E) + eta_g`, with `eta_g ~ N(0,T)`.
Individual morphology is `M_gj = U_g + epsilon_gj`, where
`epsilon_gj ~ N(0,R)` independently of ecology and group effects.

C and R must be positive definite; **T may be positive semidefinite**, including
zero. The joint between-group covariance is
`B = [[T+A C A', A C], [C A', C]]`.
This is a reparameterization of the original full Gaussian random-effects
covariance, with semidefinite boundary points included. It is not an extra
mixture or a biological group-count model. Rank describes estimated residual
morphology variation after accounting for ecology, not species number.

With `Y_g = mean_j M_gj`, the conditional covariance is `T + R/n_g`.
It remains positive definite when T is singular because R is positive definite
and every observed n_g is positive. A singular random-effects covariance can be
a mathematically valid model; see the [lme4 documentation](https://lme4.github.io/lme4/reference/isSingular.html).
Observation covariance, identifiability, dense GLS workspace and convergence are checked separately.
No singular residual or ecology covariance is accepted.

Fit `mu_E,C` by their Gaussian ML estimators. Profile morphology regression
coefficients by GLS. The remaining conditional negative log likelihood is

`0.5 * [sum_g {log|T+R/n_g| + r_g' inv(T+R/n_g) r_g}
         + (N-G) log|R| + trace(inv(R) W)]`,

where `r_g` is the group-mean regression residual and W the pooled within-group
morphology scatter. Group counts enter the likelihood explicitly. Original
measurement units are restored after internal centering/scaling. The raw joint
likelihood includes the group-mean Jacobian and Gaussian constants; independent
dense-observation checks verify it.

## Bounded optimization and acceptance

Four fixed starts initialize T/R with corrected moments, smaller/larger residual
covariances and T=0. Factor-space L-BFGS-B runs up to the declared
`max_iterations` (default 1,000), with analytic gradients and fixed numerical
log-diagonal guards for R. Each returned candidate then receives at most
`solver.max_polish_iterations` projected-covariance steps (default 2,000), with
at most 40 backtracks per step and sufficient-decrease checks. All starts and
stages are recorded. This fixed procedure is used for every observed dataset;
it is not an outcome-triggered retry policy.

Factor gradients alone cannot certify a boundary optimum: when a factor is
zero, its gradient can vanish despite a beneficial feasible covariance
direction. Acceptance therefore requires both

`max_abs(T - projection_PSD(T - gradient_T)) <= gradient_tol`

and a small R score, evaluated on standardized negative log likelihood per
group. The default tolerance is 1e-5. R and C must pass their prespecified
positive-definiteness/rank guards. Every accepted candidate is checked again
after numerical zero canonicalization; select the lowest likelihood among
accepted starts. Optimizer status alone is insufficient. A candidate that
reaches a factor-stage limit can be accepted only after the bounded continuation
and covariance stationarity checks establish stationarity. A global optimum is
not certified.

PSD projection occurs **inside constrained optimization**. No ridge, shrinkage,
positive variance floor or post-hoc repair of a failed fit is supplied. Only
machine-roundoff directions of a standardized symmetric root are canonicalized
to zero, at `64 * eps * dimension * max(spectral_norm,1)`. This is separate
from the declared numerical rank threshold 1e-8. Small positive eigenvalues
above roundoff remain in the fitted sampling law even when classified as near
boundary. Unresolved optimization, invalid observation covariance and
non-identifiability still produce failure records and withhold p-values.

## Versioned simulation and inference

Schema-2 artifacts retain the conditional covariance, regression, ecology
covariance, numerical rank/eigenvalues, cone residual, every optimization stage,
and explicit joint/within sampling factors. New raw observations are generated
from these factors, including valid singular group effects, with the observed
counts and identities fixed. Factors are constructed in standardized coordinates
and rescaled, avoiding measurement-unit-dependent variance truncation.

Schema 1 continues using the original joint Cholesky sampler and seed namespace;
schema 2 has a distinct recorded namespace and conditional factor recipe. The
same integer seed therefore does not imply a shared bootstrap dataset across
versions. Raw observed data and point-clustering settings are held identical in
the paired benchmark; bootstrap p-value differences include Monte Carlo variation.

Each bootstrap dataset reruns the complete existing pipeline: raw ingestion,
feature selection/filtering, scaling, source-view PCA, explicit morphology score
averaging, view normalization/weighting, fusion, joint PCA and all GMM K
candidates. Keep global BIC improvement, B=99, alpha=0.05, plus-one upper tails,
ties and the same fixed analysis seed. Stability remains separate. Any requested
failed draw withholds the primary p-value and preserves all denominators.

P-values are **approximate plug-in Gaussian-model departure measures**.
Boundary fits are nonregular; ordinary bootstrap validity cannot be assumed at
all boundary points. [Andrews (2000)](https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-0262.00114)
gives a counterexample for boundary estimation. It does not directly analyze
this BIC clustering statistic. Neither that paper nor variance-component LRT
results establish the validity of Super Deli's particular statistic. This
checkpoint adds targeted simulation checks, not a proof or uniform size
correction. No asymptotic chi-square, Wald, exact-rank, general nominal-level or
species-boundary guarantee is claimed. Conditional binomial intervals cover
finite simulation uncertainty, not fitted-parameter or misspecification error.

## Prespecified validation

The original 100 observed datasets are regenerated byte-for-byte: 25 each from
independent and correlated hierarchical Gaussian nulls, a complementary
four-group alternative and heavy-tail stress. A separate 75 datasets contain
25 each of zero, rank-two and weak full-rank conditional morphology covariance.
There are **17,500 planned clustering fits**: 175 observed plus 99 bootstrap
draws each. Retain every failure and report test availability.

All cases use 64 groups, independent uniform 1–4 morphology counts, one ecology
vector/group, four features/view, two PCs/view and jointly, equal weights,
K=1..4, minimum hard size 5, 10 GMM starts, model seed 71323 and master seed
20261024. Original case names preserve observed seeds. New case names yield
separate streams. The old raw laws and input bytes are unchanged.

For each fresh null, ecology has mean zero and covariance I4, R=0.75 I4, and
A has diagonal 0.6 plus 0.2 in the next cyclic column. T uses a fixed normalized
4-by-4 Hadamard basis with eigenvalues:

| Case | Conditional morphology eigenvalues |
| --- | --- |
| `zero_conditional_gaussian` | 0, 0, 0, 0 |
| `rank_two_gaussian` | 0.8, 0.4, 0, 0 |
| `weak_conditional_gaussian` | 0.8, 0.4, 0.02, 0.005 |

These are continuous populations without generating clusters. Estimated rank
need not equal generating rank at finite sample sizes. Gaussian-null rejection
rates and their Wilson intervals concern these specific laws. Heavy-tail
rejection is model-misspecification sensitivity, not species/clustering type-I
error; alternative rejection is not proof of recovering four groups.

## Read-only audit and paired comparison

```bash
# Run each of the seven case names into a separate new directory.
super-deli benchmark-grouped-calibration examples/boundary-calibration/study.json \
  --case zero_conditional_gaussian --output-dir /path/to/study/zero_conditional_gaussian
super-deli analyze-grouped-calibration /path/to/study --output-dir /path/to/new-audit
super-deli compare-grouped-calibration /path/to/legacy-study \
  --current-study /path/to/study --output-dir /path/to/new-comparison
```

The auditor regenerates all raw inputs, checks archived bytes and identities,
and independently evaluates dense raw-observation likelihoods and covariance/
mean scores. Chain-rule checks cover the conditional covariance cone, regression,
ecological covariance and within covariance. It validates sampling factors,
rank diagnostics, selected candidate, fusion/projection, selected GMM likelihood/
BIC/membership, K search and inference. No optimizer or clustering is rerun.
View PCAs and nonselected likelihoods are preserved/checksummed, not independently
refitted. Outer failures retain ledger/error verification of partial artifacts.

The paired comparator requires identical observed design/seeds/point settings,
checks all raw input bytes and outcome hashes, and exports availability changes,
point statistics, p-values, ranks and null likelihoods for the original cohorts.
It is descriptive: a higher numerical likelihood or more available tests does
not establish better inferential calibration. Outputs use exclusive new
directories and both studies remain unchanged.
