# Joint genomic–phenotypic empirical fitted-null calibration

```bash
python -m pip install '.[genomics,clustering]'
super-deli calibrate-joint-integrative /path/to/dataset.json \
  --calibration-config examples/joint-calibration/calibration.json \
  --output-dir /path/to/new-calibration
```

Python 3.11/3.12 are supported. This workflow fits a raw joint law before PCA,
regenerates all selected raw modalities, and applies the complete existing
integrative PCA/full-covariance GMM pipeline to every draw. It extends the
independent-block genotype-profile model with a conditional multivariate Gaussian
phenotype model. Earlier genomic, continuous and grouped calibrations retain
their distinct hypotheses and commands.

## Input and scientific contract

Select one genomic view first, followed by continuous morphology and/or ecology.
Every selected view must directly measure exactly the same unrelated independent
specimens, matched through explicit case-sensitive IDs. Source row orders and
observation IDs may differ and are preserved. At least one phenotypic view is
required. One ecology row per locality copied onto multiple specimens is not an
independent specimen measurement and is outside this model. Group aggregation,
unmatched individuals, related/repeated measures, spatial dependence, counts,
categorical traits, shape and missing calls/features/views require other models.
Unselected modalities/features do not enter the fitted law or generated inputs.

Genomic input is complete diploid biallelic data read through SNPio's VCF,
PHYLIP, STRUCTURE or GENEPOP readers into `GenotypeData`; popmaps are optional.
Use a fixed panel without unmodeled sample-dependent ascertainment, a prespecified
map covering every original source locus, and at least two assumed-independent
blocks. Within-block associations are arbitrary; HWE and phase are not imposed.
All existing ploidy, unsupported-call and locus-identity checks apply.

Explicit declarations assert these assumptions, constant Gaussian conditional
residual covariance and prespecified predictor blocks. They are not conclusions
inferred by the software. Predictors are an intercept and arithmetic mean raw
dosages in each declared `predictor_blocks` entry. They are not selected by trait
association, clustering, PCA, stability or the resulting p-values. Unequal block
widths are permitted by the calibration API; the validation study fixes eight
SNPs per block and the first four predictor blocks.

SNPio's eligible dosage coding is oriented using the bases in its ingested
`GenotypeData`: copies of the lexicographically larger observed base, with
monomorphic calls zero. Original base pairs, source indices and any reversal of
SNPio's dosage polarity are retained. This avoids choosing predictor polarity
from sample allele frequencies. The source allele layout is frozen for a test;
canonical bootstrap PHYLIP files use A/R/G to encode its 0/1/2 values. Point PCA
can reverse whole-locus polarity without changing the underlying calls. No
unseen block profile is introduced.

## Joint law and maximum likelihood

For specimen i, genotype vector G_i, phenotypic vector Y_i concatenated across
selected continuous views, and fixed raw design row X(G_i), the declared law is

`P(G_i, Y_i) = [product_b q_b(G_ib)] Normal(Y_i; X(G_i) B, Sigma)`.

Rows are independent. Conditional residuals are independent of the entire
genotype vector, with a constant positive-definite full covariance Sigma.
Cross-view residual covariance and fitted linear association with the declared
genomic summaries are retained. The conditional direction expresses an
association model and makes no causal claim about genotype, ecology or morphology.
It does not preserve arbitrary genomic–phenotypic dependence, nonlinear means,
heteroscedasticity, kinship or spatial structure. Even with a single conditional
Gaussian, the marginal phenotypic distribution can be non-Gaussian or multimodal.
This is a specific joint-law departure test, not a universal non-clustering null.

Likelihood factorization permits separate maximum-likelihood estimates:

- `q_hat_b(g) = observed_count_b(g) / N`, retaining observed profiles only.
- `B_hat = argmin_B ||Y - X B||_F^2`, using SVD least squares.
- `Sigma_hat = (Y - X B_hat).T @ (Y - X B_hat) / N`.

Fit before filtering, scaling or PCA. Require full column rank of the declared
raw design and enough residual dimensions (`N-rank(X) >= number of selected
phenotypic features`). Assess design rank after column-norm scaling; assess
residual rank using the residual correlation matrix to avoid dependence on trait
units. Reject numerically singular, nonfinite or underidentified fits. No null
ridge, covariance floor, shrinkage, smoothing or automatic predictor removal is
applied. GMM regularization belongs to the separate clustering statistic.

Archive raw data, coefficients, full residual covariance, design, categorical
profile counts/probabilities, sparse-support diagnostics, layout, Gaussian and
categorical ordered-row likelihoods and rank/degree-of-freedom diagnostics.

## Draws, statistic and uncertainty

Fit once per observed dataset. Keep fitted parameters fixed for all 99 fresh
bootstrap draws. Independent recorded child streams draw one original donor
per new specimen/block, then independent Gaussian phenotype residuals. Construct
each phenotype conditional mean from that draw's new genotype vector. Donor
indices are archived; a finite draw need not retain exact original frequencies,
LD or regression coefficients. No null refit inside a draw is required by the
chosen statistic.

Write new raw PHYLIP and CSV inputs preserving selected feature names, units,
sample/specimen links and observation orders. Rerun SNPio ingestion/encoding,
feature filtering, Patterson genomic scaling, tabular scaling, all view PCAs,
weight normalization/fusion, joint PCA and every prespecified GMM candidate.
The point analysis uses the same settings and model seed in observations and
draws. The integrative point pipeline uses full-covariance Gaussian mixtures;
the genomic-only diagonal-mixture baseline remains separate. Its score
preprocessing does not require a linkage model when stability is disabled.

Use global `T = BIC(K=1) - min_eligible BIC(K=1..Kmax)`, including the full fixed
K search. Count ties in the upper tail and report
`p=(1 + number(T_draw >= T_observed))/(B+1)`; default B=99, alpha=0.05 and
`p <= alpha` gives minimum p=0.01. This fitted-model bootstrap is approximate,
not an exact exchangeable-rank or permutation test. Estimated support, regression,
residual covariance, block map and model misspecification affect its behavior.
Choosing which dependence a clustering null retains is substantive; see
[Hennig and Lin (2015)](https://arxiv.org/abs/1502.02574).

Conditional Clopper–Pearson tail intervals measure Monte Carlo uncertainty given
observed T and fitted parameters; they exclude null-estimation and model
uncertainty. Any failed requested draw withholds the primary p-value, decision
and interval; failures are never replaced. Failure-outcome bounds are retained
and are not confidence intervals. Failed observed searches leave all draws
unattempted. Null-fitting and archive/workspace failures retain explicit attempt
counts. Existing inputs/outputs are never overwritten. Bootstrap stability must
be run separately with an ordinary integrative analysis.

Neither rejection nor non-rejection provides species-boundary confidence,
selected-K significance, cluster-specific significance, or a general size
or power guarantee. Continuous clines and heavy tails can reject without species
boundaries. Sparse empirical block support remains a material limitation.

## Fixed validation study

The [study declaration](../examples/joint-calibration/study.json) fixes 25 datasets
per case, N=64, 128 SNPs in 16 declared eight-SNP blocks, predictors from blocks
0–3, four features per morphology/ecology view, two PCs per view and jointly,
equal view weights, K=1..4, minimum component size 5, 10 EM starts, 99 null draws,
alpha=0.05 and master seed 20261028. Model seed is 71323. There are **125 observed
datasets and 12,500 planned full-pipeline fits**, with independent per-dataset
fitted nulls and no shared bootstrap bank, retries, adaptive stopping or tuning.

| Case | Raw law | Interpretation |
| --- | --- | --- |
| Zero association | Independent moderate-LD genotype blocks; Gaussian phenotypes with zero genomic slopes and correlated residuals | Joint-law null |
| Linear association | Same genotype law; nonzero fixed slopes on four block means; full correlated Gaussian residuals | Joint-law null preserving association |
| Complementary | Four balanced groups; first binary group axis shifts genomic frequencies and morphology, second shifts ecology | Constructed alternative |
| Heavy tailed | Linear-association law with elliptical t3 residuals | Conditional Gaussian misspecification stress |
| Cline | One continuous latent coordinate drives allele frequencies across blocks, with linear conditional phenotypes | Between-block independence misspecification stress |

The moderate-LD law reuses #48's generator: independent blocks, two haplotypes,
within-block copy probability 0.75 and refresh frequencies 0.15..0.5. Slopes,
residual AR(1) correlation 0.5, trait scales 0.5..1.5 and means -2..2 are fixed
in the versioned simulator. The cline uses uniform(-1,1) coordinates and locus
slopes ±3. Truth labels remain outside fitted inputs; stress cases have no
discrete generating groups. The alternative is deliberately strong and the K
ceiling matches its constructed group count.

```bash
mkdir /path/to/joint-study
for case in zero_association linear_association complementary heavy_tailed cline; do
  super-deli benchmark-joint-calibration examples/joint-calibration/study.json \
    --case "$case" --output-dir "/path/to/joint-study/$case"
done
super-deli analyze-joint-calibration /path/to/joint-study \
  --output-dir /path/to/new-audit
```

Four concurrent case jobs may each use one numerical thread. The separate smoke
configuration is an installation check. Small full-settings preflight runs use
a separate master seed and do not replace main observations.

## Audit and evidence

A read-only audit regenerates observed inputs, every null genotype/phenotype draw
and donor matrix; verifies all archived bytes; independently counts categorical
profiles and evaluates their likelihood; uses QR regression and dense Gaussian
log densities to check SVD coefficients, residual covariance and joint likelihood;
and reconstructs source identities, fusion/projection, selected full-GMM
likelihood/BIC/membership, K selection, seeds, tails and fit denominators without
clustering refits. View PCA and nonselected likelihoods are retained/checksummed,
not independently refitted. Outer failures retain counts without a complete
numerical audit of partial archives.

Reports include availability, per-case rejection rates, pointwise Wilson
intervals and conditional Monte Carlo uncertainty. Twenty-five datasets per case
cannot establish precise or general 5% error control. Simulated generating
partitions are not empirical species ground truth. Local artifacts preserve full
provenance; public metadata should contain versions, relative paths and hashes.
