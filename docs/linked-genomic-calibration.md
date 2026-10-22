# Linkage-aware empirical genomic fitted-null calibration

```bash
python -m pip install '.[genomics,baseline]'
super-deli calibrate-linked-genomic /path/to/dataset.json \
  --calibration-config examples/linked-calibration/calibration.json \
  --output-dir /path/to/new-calibration
```

Python 3.11/3.12 remain supported. This is a separate empirical block-genotype
null for the genomic PCA/diagonal-GMM baseline. The pooled unlinked HWE workflow
from #47 remains available, including its declared missing-data policy.

## Scientific and input contract

Use complete diploid biallelic genotypes, one source sample per unrelated
independent specimen, and one specimen-level genomic modality. VCF, PHYLIP,
STRUCTURE and GENEPOP are read through SNPio into `GenotypeData`, using the same
isolated ingestion/encoding pathway as the existing baseline. Popmaps remain
optional (`None`); population labels never enter the model. VCF called ploidy,
partial missing calls and eligible alleles retain the existing checks.
Non-VCF diploidy requires an explicit truthful declaration. Missing calls,
unsupported loci and repeated specimens are rejected; no complete-case dropping
or preliminary imputation is supplied by this null.

The calibration JSON contains `block_ids`, one ID per **original source locus**
in SNPio order, before filtering. Equal IDs identify one block; each locus has
exactly one entry, with at least two distinct blocks. Noncontiguous entries are
allowed, so the user must supply the scientifically appropriate mapping. The
map is immutable configuration, serialized and hashed with the full declaration;
it is not learned or tuned from these specimens. No physical-distance heuristic
or automatic LD-block estimation is supplied. The fixed panel must have no
unmodeled sample-dependent ascertainment.

Users explicitly assume unrelated independent specimens, independent declared
blocks, a fixed panel and prespecified block definitions. Approximate LD blocks
are an analysis assumption, not a proof of independence; see
[Berisa and Pickrell (2016)](https://doi.org/10.1093/bioinformatics/btv546).
Residual LD across declared boundaries remains a model violation. Kinship,
continuous structure, geographic sampling and ascertainment need their own
models before broader interpretation.

## Empirical fitted law

For block b, let `G_ib` be the entire vector of diploid 0/1/2 dosages for
specimen i at that block's original loci. Fit

`q_hat_b(g) = count_i(G_ib = g) / N`,

`P_hat(G_i) = product_b q_hat_b(G_ib)`.

Fit before filtering or PCA, retaining observed monomorphic profiles exactly.
The ordered-row negative log likelihood is
`-sum_b sum_g count_b(g) log(q_hat_b(g))`; no multinomial combinatorial constant
is included because the original ordered specimen observations are modeled.

This law permits arbitrary within-block genotype associations and heterozygosity.
It does **not impose HWE**, infer phase or fit evolutionary haplotype/recombination
parameters. Singleton blocks therefore fit empirical three-category genotype
probabilities, rather than the pooled Binomial(2,p) law from #47. Comparisons
between those workflows concern different null hypotheses.

Each null specimen draws an original specimen donor **independently for each
block**, uniformly with replacement. Copy that donor's whole block profile.
Repeated profiles are naturally weighted by their observed counts. Fitted
within-block joint distributions are preserved in the sampling law; a finite
draw need not preserve exact observed profile counts, allele frequencies or LD.
Cross-block specimen association is removed under the assumed product law.
This differs from the baseline's stability bootstrap, which resamples genomic
columns/blocks while retaining specimen identities.

Record every block's original indices, category profiles/counts/probabilities,
profile membership and sparse-support diagnostics. Each draw archives its
specimen-by-block donor-index matrix, indexed by original source sample order,
and a donor-layout file. Canonical PHYLIP outputs preserve sample IDs, specimen
links and original locus order. SNPio can reverse a whole locus's allele coding;
the audit permits only that reversal and checks donor copies in the fitted
orientation. Source files and original annotations remain staged and hashed.

There is no smoothing, pseudo-count or generation of unseen block profiles.
Large/high-diversity blocks can have sparse empirical support, even when every
specimen has a different profile. Diagnostics expose this limitation rather
than selecting block sizes from the resulting p-values. Missing-data block
models and profile-support regularization are later work.

## Full-pipeline inference and failures

Use 99 raw null draws and fixed settings/seeds. Each draw reruns SNPio
reading/encoding, locus filtering, Patterson scaling, fixed PCA and the full
diagonal-GMM candidate search. The observed data follow the identical pipeline.
The map is passed to the baseline with `assume_unlinked=false`; stability is
explicitly disabled in calibration. The global statistic is

`T = BIC(K=1) - min_eligible BIC(K=1..max_k)`.

The complete prespecified K search is included in the null distribution. Count
ties with `T_null >= T_observed`; report `(1 + exceedances)/(B+1)` and compare
to alpha=0.05. This is an **approximate empirical fitted-null bootstrap**, not
an exact conditional permutation test or an exact exchangeable-rank test.
Frequency/support estimation, sparse profiles, block choice and departures from
the independent-block model can affect finite-sample size.

Conditional 95% Clopper–Pearson intervals measure Monte Carlo uncertainty in
the fitted-null tail, given observed T, block map and fitted profile law. They
exclude profile-estimation and model uncertainty and do not quantify species
confidence. A failed draw withholds the primary p-value/decision, retains
failure-outcome bounds and is never replaced. A failed observed fit leaves the
null fits unattempted. Null-fit, workspace and archive-budget failures preserve
attempt counts and diagnostics. Existing outputs and inputs are never overwritten.

A rejection concerns a departure detectable by this statistic under this block
law. It supplies no species-boundary, selected-K or cluster-specific significance.
A non-rejection does not validate block independence or species number.
Continuous genetic differentiation can produce apparent clusters; see
[Bradburd et al. (2018)](https://doi.org/10.1534/genetics.118.301333).

## Prespecified validation and paired comparison

```bash
mkdir /path/to/linked-study
for case in common_hwe four_groups linked_stress cline_stress moderate_ld split_block_stress; do
  super-deli benchmark-linked-calibration examples/linked-calibration/study.json \
    --case "$case" --output-dir "/path/to/linked-study/$case"
done
super-deli analyze-linked-calibration /path/to/linked-study \
  --output-dir /path/to/new-linked-audit
super-deli compare-genomic-calibration /path/to/pooled-hwe-study \
  --current-study /path/to/linked-study --output-dir /path/to/new-comparison
```

Use 25 independent datasets/case, 64 specimens, 128 loci, 16 declared eight-SNP
blocks, two PCs, K=1..4, minimum hard cluster size 5, 10 EM starts, 99 null draws
and alpha=0.05: **150 datasets and 15,000 planned clustering fits**. Study master
seed 20261026 retains the exact 100 complete observations from #47. All new null
streams use a distinct versioned namespace. No shared null bank or post-result
tuning is used. Truth labels remain separate from fitted inputs.

| Case | Generating law | Role |
| --- | --- | --- |
| Common HWE | #47's independent common-frequency SNPs | Block-law null; potentially sparse profile support |
| Perfect-LD blocks (`linked_stress`) | #47's 16 independent genotypes copied to eight loci each | Block-law null with exact within-block linkage |
| Four groups | #47's four balanced groups, two log-odds axes | Constructed alternative |
| Continuous cline | #47's uniform latent coordinate and locus slopes ±3 | Continuous-structure misspecification stress |
| Moderate LD | 16 independent eight-SNP blocks; two independent haplotypes/specimen, with adjacent haplotype state copied with probability 0.75, otherwise refreshed from locus-specific Bernoulli frequencies | New block-law null |
| Split-block stress | Eight independent genotype blocks copied to 16 SNPs each, deliberately split into the declared eight-SNP blocks | Residual between-block LD misspecification stress |

The moderate-LD construction starts each haplotype independently at each block;
copy/refresh transitions only occur within that block. The split-block stress
has no discrete generating groups. Its rejection, if present, measures map
misspecification sensitivity. Non-rejection would not demonstrate general
robustness to incorrect boundaries. The rare/missing case from #47 is excluded
from this complete-data checkpoint.

The paired comparator requires equal study dimensions, seeds and point settings,
allowing only the declared linked/unlinked and error/mean missing-policy changes
on complete observations. It verifies raw PHYLIP/dataset bytes, saved values
hashes and observed statistic/K identity, and reports availability/p-value
changes. The different fitted nulls and draw namespaces prevent interpreting
p-value differences as a common-random-bank estimator comparison.

The read-only audit regenerates observations, draws and donor files, checks
all archive hashes, evaluates profile MLEs and ordered categorical likelihood
independently with counted tuples, verifies block support/map diagnostics, and
reuses the genomic PCA eigensystem/projection and selected-GMM likelihood audit.
It reconstructs all seeds, selection, tail probabilities, evidence and fit
counts without ingestion or clustering refits. Nonselected likelihoods are
retained/checksummed; outer failures retain counts without a complete numerical
audit. Rejection rates include availability and pointwise Wilson intervals;
25 datasets/case cannot establish precise or general 5% error control.
