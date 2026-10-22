# Four-group integrative recovery diagnostics

```bash
super-deli diagnose-integrative-recovery examples/recovery-diagnostics/study.json \
  --source-artifacts /path/to/preserved/joint-null-20261021 \
  --output-dir /path/to/new/recovery-study --workers 4
super-deli analyze-integrative-recovery /path/to/new/recovery-study \
  --output-dir /path/to/new/recovery-audit
```

Install `super-deli[genomics,clustering]` on Python 3.11 or 3.12. Both destinations
must be new and separate from the source artifacts. The published configuration
pins the source study declaration, its complete study-file inventory and the
frozen module inventory. Full source artifacts are retained locally; the compact
public result files cannot substitute for the archived raw data and models.

This checkpoint diagnoses the constructed complementary alternative from the
joint genomic–phenotypic study. It changes no production model, default,
stability result or significance calculation. It does not establish biological
species boundaries or a general recovery guarantee.

## Frozen design

There are three cohorts, each with 25 independently generated datasets: the
archived N=64 observations, fresh N=64 observations, and fresh N=256 observations.
The fresh cohorts use master seed 20261029 and distinct cohort-derived streams;
N=64 and N=256 are independent observations, not nested samples. Every dataset
has balanced generating labels 0–3, 128 diploid loci, four morphology features
and four ecology features. Genomics and morphology encode axis A (labels // 2);
ecology encodes axis B (labels % 2). Correlated, heteroscedastic phenotype noise
is inherited unchanged from the frozen generator. Truth labels are separate
evaluation artifacts and never enter an unsupervised pipeline declaration.

The seven point-analysis configurations are fixed before fresh generation:

| Configuration | Change from source baseline |
| --- | --- |
| baseline | None |
| pcs_3 | Three joint PCs |
| pcs_4 | Four joint PCs |
| ecology_0.5 | Ecology weight 0.5 |
| ecology_2 | Ecology weight 2 |
| ecology_4 | Ecology weight 4 |
| starts_100 | 100 EM initializations |

All other controls remain fixed: two PCs per view, equal baseline view weights,
full-covariance GMM K=1..4, minimum hard component size five, ten baseline EM
initializations, seed 71323, maximum 500 EM iterations, tolerance 0.001 and
covariance regularization 1e-6. All 525 points rerun raw SNPio ingestion,
filtering/scaling, view PCA, profile normalization, weighted fusion, joint PCA
and the complete GMM search. No bootstrap or significance draws are requested.
Up to four separate processes run these single-threaded fits.

The workflow first checks source file hashes, regenerates archived raw inputs
and truth, verifies identifiers and audits archived observed-point models.
Scientific source modules must match the pinned historical inventory; only the
package version and CLI dispatch may differ. Parsed source configurations are
compared after default expansion. The archived baseline must reproduce
candidate statuses/sizes, BICs, selected K, scores and labels within declared
numerical tolerances before its ablations are interpreted.

## Optimization diagnostic

Each baseline score space also receives one forced K=4 EM fit initialized using
the four generating groups: weights are their proportions, means are group
score means, and covariance estimates are within-group maximum-likelihood
covariances plus the existing regularization. EM then proceeds unrestricted
with one initialization and the baseline convergence settings. These 75 fits
use truth information and are reported separately. They are neither an
unsupervised performance estimate nor an alternative selected model. A failure
is retained; a failed baseline leaves its truth-initialized fit unattempted.

## Measures and comparisons

Adjusted Rand agreement compares the selected partition with generating labels.
Axis homogeneity describes how much of each binary generating axis is explained
by the selected partition. In-sample linear R² for each generating axis is
reported for view PCs, the six-dimensional fused scores and joint PCs. These
are descriptive signal-retention measures, not held-out accuracy or p-values.
Joint PCA variance fractions and squared loadings summed within each view
locate which views contribute to retained components. PCA signs are arbitrary.

For N observations and d scores, a full-covariance K-component GMM has
`p = K * (d + d*(d+1)/2) + K - 1` parameters and
`BIC = -2*log_likelihood + p*log(N)`. Candidate tables separate log-likelihood
gains from extra penalties **within each fixed score space**. Nonselected
likelihoods are derived algebraically from preserved BIC logs, rather than
independently evaluated from missing component arrays. Absolute BIC across
different PC dimensions or differently weighted/projected score spaces is
not compared. More EM starts can select a higher-likelihood candidate that
violates the hard size constraint; eligibility must therefore be reported
alongside BIC and ARI. The production behavior is preserved.

Within-dataset configuration comparisons are descriptive paired comparisons;
the two fresh sample sizes have independent seeds. Pointwise Wilson intervals
for selecting K=4 describe binomial simulation uncertainty and do not adjust
for the seven-configuration grid. There is no post-hoc winning configuration,
default change, hypothesis-test rerun, replacement dataset or replacement fit.
Any method change requires a separate checkpoint and new validation.

The scikit-learn documentation describes the
[GMM BIC search](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html)
and [initialization controls](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html).
The frozen numerical run uses scikit-learn 1.6.1; current documentation may
describe a later version.

## Evidence and audit boundaries

The study saves every input, generating label, configuration, source snapshot,
candidate ledger, score/profile/PCA array, selected mixture and truth-initialized
mixture. Failures remain in denominators without retries. A complete relative
file inventory closes the study. The separate auditor regenerates raw inputs,
checks identities/settings/seeds/ledgers and baseline replay, reconstructs
fusion and PCA projection, and evaluates the selected mixture's dense
likelihood/BIC/membership. It independently reconstructs ARI from contingency
counts, axis recovery with a pseudoinverse and truth-initialization parameters
and fitted likelihoods. It does not refit any clustering model. View PCA fits,
nonselected component arrays and convergence histories are not independently
refitted; nonselected likelihoods remain BIC-derived logs. Full frozen runtime,
source and raw artifacts remain local; portable tables and hashes are published.

Run the small installed-wheel smoke independently with:

```bash
python examples/recovery-diagnostics/smoke.py \
  --joint-study examples/joint-calibration/smoke.json \
  --output-dir /tmp/new-recovery-smoke
```

It constructs its own small current-runtime source archive, then calls both
installed CLIs using two worker processes. Python 3.11/3.12 CI performs this
outside the source checkout. The smoke is a software check, not scientific
validation of the 75-dataset study.
