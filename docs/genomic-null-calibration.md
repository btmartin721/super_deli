# Empirical fitted-null calibration for genomic data

```bash
python -m pip install '.[genomics,baseline]'
super-deli calibrate-genomic /path/to/dataset.json \
  --calibration-config examples/genomic-calibration/calibration.json \
  --output-dir /path/to/new-calibration
```

Python 3.11 and 3.12 remain supported. This checkpoint calibrates the existing
**genomic-only** PCA/diagonal-GMM baseline against a fitted pooled Hardy–Weinberg
(HWE) null. It does not define a joint genomic–phenotypic fitted null.

## Input and scientific contract

Use one specimen-level genomic modality, with one source sample per unrelated
biological specimen. Read VCF, PHYLIP, STRUCTURE or GENEPOP through SNPio into a
`GenotypeData` object, using its public readers and encoder. Popmaps remain
optional (`None`); neither population labels nor any generating truth enter the
fit. Observation links join by sample identity rather than declaration order.

The declaration must explicitly assume independent unrelated diploid specimens,
unlinked loci, HWE, a fixed panel without sample-dependent ascertainment, and
non-informative missingness. These are scientific assumptions, not conclusions
established by software. SNPio-supported file formats do not make every variant
type eligible: this calibration requires supported biallelic SNP calls only.
VCF called ploidy/alleles and partial missing calls are checked by the existing
encoder. Non-VCF diploidy requires a truthful declaration. Unsupported loci and
entirely missing loci are rejected, with diagnostics retained. Canonical draws
require whitespace-free sample IDs. Budget guards limit numerical workspace and
uncompressed per-test archives.

The panel assumption excludes selecting loci by a variant/MAF threshold computed
from these same specimens without modeling that selection. A typical
variant-only VCF may violate it. Externally ascertained markers are not
necessarily representative of an unascertained genome either; interpretation
is conditional on the explicitly assumed panel law. This checkpoint does not
estimate ascertainment, LD, inbreeding, kinship, admixture or spatial processes.

## Fitted raw-genotype null

For called dosage `G_ij` at original source locus j,

`G_ij ~ Binomial(2, p_j)` independently across specimens and loci,

`p_hat_j = sum_i_called G_ij / (2 * number_called_j)`.

Fit these maximum-likelihood frequencies **before** locus filtering, imputation,
scaling or PCA. The observed missing-call mask is held fixed; no latent missing
genotype is used to estimate a frequency. Observed monomorphic loci retain exact
0/1 frequency boundaries, with no pseudo-count, frequency floor or redraw to
force a variable locus. A draw can lose variable loci or become unfit; it is
retained as a failure. SNPio may reverse an entire locus's allele coding. That
reversal leaves Patterson PCA geometry unchanged and is explicitly checked in
the independent audit.

Raw draws are written as canonical PHYLIP/IUPAC `A/R/G/N`, carrying sample IDs,
specimen links and the original source locus order. The saved layout maps every
draw column to its zero-based original locus index. These files do not reproduce
VCF annotations, phasing, quality scores or original allele names; those remain
in staged observed inputs and their provenance.

## Full-pipeline inference

For observed data and each of 99 independently seeded raw null draws, rerun SNPio
reading/encoding, missing/zero-variance locus handling, declared mean imputation,
Patterson scaling, fixed-dimensional PCA and every feasible diagonal-GMM
candidate K=1..4. Fixed model settings and the same prespecified analysis seed
are used for every fit. The statistic is

`T = BIC(K=1) - min_eligible BIC(K=1..max_k)`.

The complete search belongs to this one global statistic. Hard minimum cluster
size, EM convergence, regularization and candidate admissibility retain the
baseline's behavior. A numerical candidate failure fails that pipeline fit;
small hard components are ineligible under the prespecified size rule.

Count ties in the upper tail. Report `(1 + # {T_null >= T_observed}) / (B+1)` and
compare to the declared alpha (0.05 in the study). Because frequencies are fitted
from the observed data, this is an **approximate plug-in parametric bootstrap**,
not an exact exchangeable-rank test. Boundary/rare frequencies and the number of
fitted nuisance parameters can affect its finite-sample behavior. Repeating the
full preprocessing/PCA search does not restore an exact size guarantee.

Report a 95% Clopper–Pearson interval for the fitted-null tail probability,
conditional on observed T, estimated frequencies and missing mask. It measures
Monte Carlo uncertainty, not frequency-estimation uncertainty or species
confidence. Any failed draw withholds the primary p-value and decision, retains
failed-outcome bounds, and is not replaced. A failed observed fit leaves all
null fits unattempted. Null-fit/archive failures preserve attempt counts in
`failure.json`. No output or source input is overwritten. Stability bootstrap
is disabled here and remains a separate analysis.

A rejection establishes departure detectable by this statistic under this
particular null. LD, continuous structure, relatedness, technical artifacts and
misspecified missingness/ascertainment can cause departure without discrete
species. Non-rejection does not establish panmixia, HWE or a single species.
The distinction between population structure and a discrete clustering model is
also discussed by [Patterson et al. (2006)](https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.0020190).

## Prespecified validation

```bash
mkdir /path/to/genomic-study
for case in common_hwe rare_missing_hwe four_groups linked_stress cline_stress; do
  super-deli benchmark-genomic-calibration examples/genomic-calibration/study.json \
    --case "$case" --output-dir "/path/to/genomic-study/$case"
done
super-deli analyze-genomic-calibration /path/to/genomic-study \
  --output-dir /path/to/new-audit
```

The committed design fixes 25 independent datasets per case, 64 specimens and
128 panel loci, two PCs, K=1..4, minimum hard cluster size 5, 10 EM starts and 99
raw null draws. There are **12,500 planned clustering pipelines**; null-model
estimation ingests each observed dataset separately and is not an extra
clustering fit. No seeds/settings are selected after inspecting results.

| Case | Raw generating law | Interpretation |
| --- | --- | --- |
| Common HWE | Independent loci, p uniform on [0.15,0.5] | Model-law rejection rate |
| Rare/missing HWE | Independent loci, p uniform on [0.01,0.1]; 10% independent missing calls | Model-law rejection rate with rare/boundary fits and fixed masks |
| Four groups | Four balanced generating groups, two binary log-odds axes of magnitude 2.5 | Power for this constructed alternative |
| Linked stress | 16 independent HWE blocks, each copied to eight perfectly linked loci | Sensitivity to linkage misspecification |
| Continuous cline | Uniform continuous latent coordinate, locus log-odds slopes ±3 | Sensitivity to continuous structure without discrete generating groups |

Locus frequencies and missingness are generated afresh per observed dataset;
there is no conditioning on observed polymorphism or post-generation MAF
selection. The cline/stress truth files carry no invented discrete biological
partition. The `smoke.json` design reduces only dataset/draw/fit counts and is
used for CLI/package checks, not the published study.

The study reuses the existing independent-dataset ledger, fit accounting,
Monte Carlo tails, rate summaries and raw-pipeline archive engine. A small
loader/point-record adapter connects genomic data without routing genotypes
through Gaussian continuous-data models. Rejection rates use available tests
with availability counts and pointwise 95% Wilson Monte Carlo intervals.
Twenty-five datasets per case give limited precision, even with zero rejections.

The read-only audit regenerates raw observations and draws, verifies every
archive checksum and seed, independently evaluates allele-frequency MLEs and
binomial likelihoods, reconstructs filtering/imputation/scaling, checks the PCA
eigensystem/projection and selected diagonal-GMM likelihood/BIC/membership, and
recalculates selection and p-values. It does not call SNPio or refit clustering.
Nonselected mixture likelihoods are retained in candidate logs rather than
independently recomputed from unexported parameters. Outer failures retain
attempt accounting without claiming a complete numerical re-audit.
