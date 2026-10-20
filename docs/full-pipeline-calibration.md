# Full-pipeline calibration under a declared reference population

```bash
python -m pip install '.[genomics,clustering]'
super-deli calibrate-reference-pipeline /path/to/observed/dataset.json \
  --calibration-config examples/pipeline-calibration/calibration.json \
  --output-dir /path/to/new-calibration
```

Python 3.11 and 3.12 are supported. Each invocation reads the original inputs
without modifying them. The output directory must be new. The explicit
`assume_declared_reference=true` asserts that the declared simulator is an
appropriate reference distribution for the supplied data. Compatibility checks
verify its dimensions, named features, landmark correspondence and sampling
design; they cannot establish the scientific suitability of that assumption.

## Hypothesis and scope

This checkpoint tests a **known raw-data reference population**, named
`four_view_continuous_population_v1`. All its parameters are declared in advance.
There is no empirical parameter estimation for the null. The global null says
that the observed raw data follow this complete simulation law, including its
sampling design. A small p-value indicates departure from that reference law.
It does not identify species, validate individual boundaries, test the selected
number of clusters or estimate the probability that an assignment is correct.

The existing `calibrate-integrative` command remains a separate approximate
fitted-Gaussian test of frozen scores. Here every null draw regenerates all four
raw modalities and reruns the entire configured point pipeline:

1. SNPio PHYLIP ingestion into `GenotypeData`, with optional popmap absent.
2. Genomic encoding, eligibility filtering, missing-cell handling and view PCA.
3. Typed morphology/ecology ingestion, configured transformations, filtering,
   imputation, scaling and view PCA.
4. Landmark-coordinate ingestion, fixed-point GPA and shape PCA.
5. Exact unit alignment or explicit locality aggregation of source PC profiles.
6. Profile centering, variance normalization, prespecified weighting, fusion,
   joint PCA and the complete configured full-covariance GMM K search.

Dimensions, weights, transformations, hyperparameters, alpha and analysis seed
are fixed before seeing outcomes. Retained features, imputation estimates, GPA,
PCA bases, aggregation profiles, normalizations and mixture parameters are
learned anew for every fit. Complete-case selection is repeated when declared.
This covers their estimation under this raw-data law. It does not cover choosing
the analysis settings after inspecting results.

## Reference law

The generator is reused unchanged from the integrative benchmark. A single
continuous population has independent target latent vectors with two normal
coordinates of SD 0.35. Each source observation adds latent normal noise of SD
0.3. Modalities share target latent variables; they are not independent views.

- Genotypes are diploid, independent binomial loci conditional on latent values.
  Baseline log-odds are independently uniform on [-0.5,0.5] per dataset/locus;
  half the loci use each latent coordinate with coefficient 0.65. PHYLIP IUPAC
  `A/R/G` encodes 0/1/2 copies. No linkage, genealogies or demographic process is
  simulated. Baseline frequencies are redrawn for every reference dataset.
- Morphology/ecology contain four continuous linear latent traits with normal
  noise of SD 1. Morphology also contains a Poisson count with log mean
  `1.8 + 0.15 * sum(source_latent)`.
- Shape contains six corresponding 2D fixed landmarks on a hexagonal template,
  latent deformation of scale 0.08 and normal coordinate noise of SD 0.02,
  followed by random rotation, uniform scale in [0.5,3] and translation in
  [-10,10] on each axis. It supplies coordinates, not learned image landmarks.
- Matched data have one observation per target per modality. Explicitly grouped
  unequal sampling independently draws 1..4 individuals per target in genomic,
  morphological and shape views. Ecology is measured once per locality.
  Source counts are redrawn for every dataset; calibration is marginal over
  those counts, not conditional on the realized observed counts.
- Optional declared cell missingness affects genomic/tabular cells; optional
  whole-view missingness removes ecological coverage. Landmarks stay complete.
  The main validation study uses complete coverage and cells. Unit tests also
  exercise imputation and complete-case alignment.

The exact versioned implementation and its SHA-256 accompany every test.
Genomic input for this reference law is generated as PHYLIP; the existing SNPio
adapter continues to support VCF, PHYLIP, STRUCTURE and GENEPOP for compatible
user inputs. This law's scientific scope remains the same regardless of format.
It supplies no empirical fitted null, spatial/related sampling, arbitrary
anatomical templates, semilandmark sliding, environmental gradients or
evolutionary species model. Additional reference laws require separate designs
and validation. Null-model choice in cluster validation is discussed by
[Hennig and Lin](https://arxiv.org/abs/1502.02574).

## Statistic, seeds and numerical uncertainty

For each raw dataset, `T = BIC(K=1) - min eligible BIC(K=1..Kmax)`. K=1 is included,
so T is nonnegative and often exactly zero. The existing minimum hard-component
size, convergence, regularization and restart rules apply. Failure of a feasible
candidate fails the search; failed candidates are not silently removed.

With B independent null datasets and b statistics at least as large as the
observed T, `p=(b+1)/(B+1)`. Ties enter the upper tail. The default B=99 gives
minimum p=0.01. The decision is `p <= alpha`, with alpha=0.05 by default.
The plus-one convention is discussed by
[Phipson and Smyth](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf).
Under the fully specified null, independent raw draws, and the same defined
deterministic analysis function applied to observed and simulated data, the
rank argument gives a conservative Monte Carlo test including ties.
It does not establish that empirical data follow this law.

The identical prespecified model seed is used for observed and all null fits.
This fixes the fitting algorithm as the same function of each raw dataset;
simulation seeds derive independently from stable reference/replicate keys.
Observed data and each dataset's null stream have separate keys in validation.
No shared null bank, retries, adaptive stopping or outcome-dependent tuning is
used. The default validation analysis seed is 71323.

The binomial Clopper-Pearson 95% interval describes the reference-tail probability
conditional on the observed T. It quantifies finite-simulation uncertainty;
it is not a species confidence interval. Any failed requested null draw makes
the primary p-value, decision and interval unavailable. Failed-outcome bounds
retain all B draws: `[(b+1)/(B+1),(b+f+1)/(B+1)]`. They are identification bounds,
not confidence intervals. A failed observed fit records B unattempted draws.
Availability can be selective, so rejection rates among successful tests alone
need not preserve unconditional error control.

Stability is separate evidence. `n_bootstrap` must be zero for this calibration
workflow to avoid nesting a stability bootstrap inside every reference fit.
Users can run and preserve a separate stability analysis with the same settings.
There is no multiplicity correction across datasets, view choices or alternative
configurations. The complete K search belongs to one prespecified global statistic.

## Prespecified validation and artifacts

The [study declaration](../examples/pipeline-calibration/study.json) uses N=64
targets, 64 loci, two PCs per view and two joint PCs, K=1..4, minimum hard size 5,
10 EM starts and one numerical thread per fit. Master seed is 20261022. There are
25 independent observed datasets in each of four cases: matched null, matched
complementary four-group alternative, grouped unequal null and grouped unequal
complementary alternative. Alternative latent group shifts have strength 3.
Every dataset has its own 99 null draws: 100 observed plus 9,900 null fits.
The alternative is a constructed partition, not empirical species truth.

```bash
super-deli benchmark-pipeline-calibration examples/pipeline-calibration/study.json \
  --case matched_null --output-dir /path/to/study/matched_null
# Repeat with matched_complementary, grouped_null, grouped_complementary.
super-deli analyze-pipeline-calibration /path/to/study \
  --output-dir /path/to/new-analysis
python scripts/plot_score_calibration.py /path/to/new-analysis \
  --reference-pipeline --output-dir /path/to/new-figures
```

Four cases can run as separate single-threaded processes. Never run SNPio case
jobs as fitting threads sharing one interpreter. The miniature smoke declaration
is for installation checks and supplies no error-rate evidence.

Each test writes `configuration.json`, `observed.json`, per-draw JSON records,
`null_replicates.json`, `evidence.json`, `summary.json`, and
`pipeline_artifacts.zip` with a complete SHA-256/size inventory. The ZIP preserves
every raw null simulation, copied observed inputs, ingestion products, alignment,
GPA/PCA/profile/fusion/mixture parameters and candidate logs. Temporary staging
directories are removed after archival. `max_archive_bytes` bounds uncompressed
artifacts per test; study storage also grows with the number of datasets.
Unexpected failures retain partial archives and a failure record.

The read-only auditor regenerates observed and null raw inputs from their seeds,
verifies archived bytes and parameters, reconstructs block normalization/fusion,
joint projection, selected GMM likelihood/BIC/membership, selection bookkeeping,
p-values and requested/attempted/successful denominators. It does not independently
refit individual view PCA/GPA or nonselected mixtures. Dataset/rate CSVs and a
full raw-file hash inventory are exported outside the study directory.

Rates weight independent observed datasets equally. Wilson 95% intervals and
Monte Carlo standard errors describe validation uncertainty; intervals are
pointwise and unadjusted. With only 25 datasets per case they are broad, especially
near zero. This bounded study cannot certify general type-I control, arbitrary
sample sizes, other reference parameters, or empirical species significance.

See the [completed validation results](full-pipeline-calibration-results.md).
