# Full-covariance significance calibration with group-mean PCA

The schema-2 grouped fitted-null workflow accepts `pca_fit="target_means"` with
`covariance_type="full"`. The GMM estimates one full covariance per component;
the existing shared (`tied`) family remains supported. Fix the family before
analysis. Source-row PCA, two joint PCs and full covariance remain defaults;
legacy full-family serialization omits the default covariance field. Historical
configuration hashes and p-values remain unchanged. Legacy schema 1 and other
target-mean significance adapters remain disabled.

## Raw null, sampling and complete refits

Reuse the [group-mean raw-null specification](groupmean-calibration.md) unchanged:
complete continuous specimen morphology with explicit population/locality links,
one directly measured ecology row/group, independent Gaussian group effects,
iid Gaussian within-morphology residuals and fixed non-informative unequal counts.
The conditional-PSD estimator permits zero/reduced-rank conditional between-group
morphology covariance; ecology and within-morphology covariance must be positive
definite. Keep the same bounded solver, convergence checks and sampling law.
The raw covariance components are distinct from GMM component covariances.

Fit the raw null once per observed dataset and hold fitted parameters/design fixed
within the test. Every independent parametric-bootstrap draw generates raw
specimen morphology and group ecology, then reruns ingestion, raw group means,
feature standardization, view PCA, projected-source aggregation, weighted fusion,
joint PCA and the full configured GMM K search. Every member/count/identity is
preserved. Each group mean has equal PCA influence; fewer specimens still imply
noisier means. Ecology is not replicated onto specimens as independent evidence.
No variance floor, adaptive restart, new raw estimator or runtime repair is added.

For n groups, d joint PCs and K components, full-family BIC uses
`q=(K-1)+K*d+K*d*(d+1)/2` parameters. The shared family counts its covariance only
once. In each fitted score space, `T=BIC(K=1)-min_eligible BIC(K=1..Kmax)` includes
the entire fixed K search, with the same convergence/size constraints for observed
and null fits. Absolute BIC is not compared across different PCA score spaces.

## Inference, failures and limitations

With B=99, use `p=(1+count(T_draw>=T_observed))/100`, including upper-tail ties,
and p<=0.05 for rejection. Report **approximate fitted-model departure p-values**.
Fitted nuisance parameters remove exact exchangeable-rank validity; covariance
boundaries are **nonregular**, with no uniform or general size guarantee. Rejection
does not establish species boundaries, selected K or membership confidence;
non-rejection does not establish Gaussianity or robust biological delimitation.

Conditional Clopper–Pearson tail intervals describe draw variation given observed
T, fitted parameters and fixed counts, excluding nuisance-estimation/numerical/
misspecification uncertainty. Across-observation rejection rates use pointwise,
unadjusted Wilson intervals. These are different uncertainty measures, neither
being species confidence. An atom at T=0/p=1 and zero plug-in Monte Carlo SE do
not imply zero uncertainty. Stability remains a separate workflow.

Failed points leave draws unattempted; any requested failed draw withholds the
primary p-value, decision and interval. Preserve all error/attempt ledgers and
failed-outcome bounds without replacing, selectively retrying or repairing draws.
Unresolved raw fits withhold inference. Bounds are not confidence intervals.
Informative counts, related/spatial groups, heterogeneous within-group residuals,
missingness, genomic/count/shape data and separately identified ecology
measurement error remain outside the null. Covariance-family/PCA/weight/feature
selection is not calibrated; no automatic family or default change is supplied.

## Frozen validation and commands

Fresh master seed **20261110**, model seed **71323**, 64 groups with independent
1–4 morphology specimens/group, one ecology row/group and four traits/view.
Fix two PCs/view, four joint PCs, equal weights, full covariance, K=1..4,
minimum size 5, 10 EM starts, existing solver budgets, 99 draws and alpha=0.05.
Use 25 fresh observations each from independent/correlated Gaussian nulls,
complementary four groups, heavy-tailed stress and zero, rank-two and weak
conditional-covariance nulls: **175 datasets and 17,500 planned pipeline fits**,
at most four one-thread jobs. Preflight seed 20261111 is separate and supplies no
replacement observations. Freeze settings/seeds/runtime before fresh generation.

The five Gaussian scenarios check the declared law, complementary groups provide
constructed power and heavy tails assess misspecification sensitivity. Twenty-five
observations per case give coarse uncertainty and no general 5% size guarantee.
Historical shared-family comparisons are **unpaired** and cannot isolate the
family effect. Truth evaluates observed partitions only after fitting; it never
chooses configurations, initializes models or enters unsupervised fitting.

```bash
super-deli calibrate-grouped-integrative /path/to/dataset.json \
  --calibration-config examples/full-groupmean-calibration/calibration.json \
  --output-dir /path/to/new-calibration
super-deli benchmark-grouped-calibration examples/full-groupmean-calibration/study.json \
  --case independent_gaussian --output-dir /path/to/new-study/independent_gaussian
# Run the six other declared scenarios into sibling directories, at most four at once.
super-deli analyze-grouped-calibration /path/to/new-study \
  --output-dir /path/to/new-audit
```

The no-refit audit verifies inventories, regenerated inputs/draws, raw dense
likelihood/constrained convergence, counts/identities, per-draw PCA moments and
eigen-equations, source projections/profiles, fusion, component-specific selected
mixture likelihood/BIC/membership, full K selection, seeds, tails and denominators.
Nonselected GMM likelihoods are retained/checksummed and not independently refitted;
convergence does not certify a global optimum. Earlier numerical failures and
all previous evidence remain preserved. Python 3.11/3.12 remain supported.
