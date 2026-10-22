# Full-pipeline calibration with shared covariance

```bash
python -m pip install '.[genomics,clustering]'
super-deli calibrate-joint-integrative /path/to/dataset.json \
  --calibration-config examples/shared-calibration/calibration.json \
  --output-dir /path/to/new-calibration
```

Python 3.11/3.12 are supported. Set `model.covariance_type` to `"tied"`
**before** examining clustering or significance results. Every component has its
own mean/weight and shares one full covariance matrix, following the
[scikit-learn GaussianMixture definition](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html).
The default remains
`"full"`; old full-family configurations and results retain their meaning.

This checkpoint enables the existing [joint genomic–phenotypic fitted
null](joint-genomic-phenotypic-calibration.md) with that prespecified clustering
statistic. It introduces no new raw null law, covariance estimator or automatic
family selection. Its separate guards still reject shared covariance in the
frozen-score, known-reference, continuous-only and grouped calibration APIs.
Shared-family calibration of shapes, missing data and grouped/unmatched sampling
requires later work.

## Inputs and null hypothesis

Use matched unrelated independent specimens, complete diploid biallelic genomic
data, and continuous individually measured morphology and/or ecology. Read
VCF, PHYLIP, STRUCTURE or GENEPOP through SNPio into `GenotypeData`; popmaps
remain optional. Require a fixed panel, a prespecified map of at least two
assumed-independent non-overlapping blocks covering all source loci, and
prespecified raw block-mean dosage predictors. Shared locality ecology copied
to several specimens, related samples, missingness, counts and shapes are outside
this null.

Fit each block's genotype-profile probabilities by count/N. Fit an intercept and
Gaussian phenotype regression on the declared raw block means, using a full
residual covariance. Fit before PCA; retain genomic–phenotypic linear association
and cross-phenotype residual covariance. Reject rank-deficient designs and
singular residual fits. The existing estimator applies no covariance floor,
profile smoothing, predictor selection or ridge. This conditional association
model has no causal interpretation. Marginal phenotypes need not be Gaussian or
unimodal. Sparse fitted profile support and unmodeled dependence remain limitations.

## Full-pipeline statistic and simulation

Hold fitted raw-null parameters fixed within a test. Draw genotype profiles
independently across specimens/blocks, compute the new raw regression predictors,
and draw Gaussian phenotype residuals. Preserve specimen/observation/locus/feature
identities and write new PHYLIP/CSV inputs for each replicate. Refit SNPio
reading/encoding, filtering/imputation settings, genomic Patterson scaling,
tabular scaling, view PCA, weight normalization, fusion, joint PCA and the full
prespecified shared-covariance GMM K search. The same model seed/settings are used
for the observed data and all draws. There is no frozen-score shortcut.

Use `T = BIC(K=1) - min_eligible BIC(K=1..Kmax)`. In d score dimensions,
the shared-family parameter count is `K*d + d*(d+1)/2 + K-1`. Exported dense
covariances repeat the common matrix for each component, but BIC counts it once.
A global upper-tail calculation includes the entire fixed K search and ties:
`p = (1 + count(T_draw >= T_observed))/(B+1)`. With B=99 the minimum is 0.01;
alpha=0.05 uses p<=alpha. When K=1 wins, T=0 and p=1.

These are **approximate fitted-model departure p-values**. Fitting nuisance
parameters prevents an exact exchangeable-rank guarantee. Conditional
Clopper–Pearson tail intervals quantify draw variation given the fitted model
and observed statistic; they exclude null-parameter, block-map and model
uncertainty. Failure-outcome bounds are retained and are not confidence intervals.
Any failed requested draw withholds the primary p-value, decision and interval;
failed observed fits leave draws unattempted. No failure is retried or replaced.
Stability bootstrap runs separately.

Fix one family/configuration in advance. Choosing full versus shared covariance,
PC counts, weights or predictors after inspecting results, then reporting that
choice's p-value, is outside this calibration. P-values do not establish species
boundaries, significance of selected K, or individual cluster support. Neither
rejection nor non-rejection supplies a general size/power guarantee. Continuous
clines and heavy tails can depart from this law without discrete species.

## Frozen validation

The [study](../examples/shared-calibration/study.json) fixes 25 fresh observations
in each of five cases: zero association, linear association, complementary four
groups, heavy-tailed residuals and a continuous cline. The first two follow the
joint null law; the last two are misspecification stresses. N=64, 128 loci in 16
eight-SNP blocks, the first four predictor blocks, four traits per phenotype
view, two PCs per view/jointly, equal weights, K=1..4, minimum size 5, 10 EM starts,
99 draws and alpha=0.05 remain fixed. Study seed 20261031 is distinct from earlier
studies; model seed is 71323. There are 125 observations and 12,500 planned
pipeline fits, without tuning, replacement observations or shared null banks.

Run each of `zero_association`, `linear_association`, `complementary`,
`heavy_tailed`, and `cline` into a separate directory under one new study root:

```bash
super-deli benchmark-joint-calibration examples/shared-calibration/study.json \
  --case zero_association --output-dir /path/to/new-study/zero_association
# Repeat for the other four cases, with no more than four concurrent jobs.
super-deli analyze-joint-calibration /path/to/new-study \
  --output-dir /path/to/new-audit
```

Each job uses one numerical thread. The no-refit audit regenerates observations,
raw draws and donor indices; independently verifies profile counts, QR regression,
Gaussian likelihoods, fusion/projection and selected mixture likelihood/BIC and
membership; and reconstructs seeds, K selection, tails, intervals and denominators.
View PCA and nonselected candidate likelihoods remain preserved/checksummed,
not independently refitted. Pointwise Wilson rejection-rate intervals across
observations are distinct from per-test Monte Carlo intervals. A smoke declaration
is provided for CI; it is not a significance validation study.

See the [completed fresh validation](shared-covariance-calibration-results.md)
for audited outcomes and limitations.
