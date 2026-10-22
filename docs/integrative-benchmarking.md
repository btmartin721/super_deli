# Simulation-based integrative benchmarks

```bash
python -m pip install -e '.[dev,clustering,genomics]'
super-deli benchmark-integrative examples/benchmarking/smoke.json \
  --output-dir /tmp/super-deli-benchmark-smoke
```

Use a new directory. Python 3.11 and 3.12 remain supported. The benchmark calls
the existing integrative workflow; it adds no estimator or tuning procedure.
Genomic arms read simulated PHYLIP through SNPio with an optional absent popmap.
Non-genomic arms need only `[clustering]`; image codecs are unnecessary.
Morphometric simulations supply coordinates; raw image decoding/detection is
outside this benchmark.

## Study contract

The design separates aims, data-generating mechanisms, targets, methods and
performance measures following [Morris, White and Crowther (2019)](https://doi.org/10.1002/sim.8086).
Independent simulated datasets are the outer experimental units. Target units
within each dataset are resampled by the existing inner bootstrap.

The aim is to test recovery of constructed partitions and the behavior of
bootstrap stability under controlled stress. This is not an evolutionary
speciation simulator, an empirical species benchmark or a species-significance
calibration study. There is no genealogy, LD, migration, spatial dependence,
selection, sex/allometry effect, measurement-annotation error or semilandmark
sliding model. Independent loci and independent target units are assumptions.

Each dataset has four balanced generating groups, with randomized group labels
independent of row identifiers. The null case has one group. A two-dimensional
target latent vector has group means at the corners `(±strength, ±strength)`
and independent Normal(0, 0.35²) target variation. Source observations add
Normal(0, 0.3²) variation. Source observations from different modalities at a
grouped target are different simulated individuals, conditionally independent
given the target vector.

| Data type | Generator |
| --- | --- |
| Genomic | Independent diploid binomial dosages. Half the loci depend on each latent axis; `p=logistic(b_l+0.65*z_axis)`, with fixed per-dataset locus logits `b_l~Uniform(-0.5,0.5)`. Dosages become A/R/G calls; missing calls become N. |
| Morphological | Four continuous measurements with means `(z1,z2,z1+0.3*z2,z2-0.2*z1)` and independent Normal(0,1) noise, plus a Poisson meristic count with mean `exp(1.8+0.15*(z1+z2))`. |
| Ecological | The four continuous measurements above, with separate observation noise; one record per target. |
| Shape | Six homologous 2D landmarks on a fixed hexagonal outline. Two deformations of size `0.08*z_axis`, Normal(0,0.02²) coordinate noise, random rotation, Uniform(0.5,3) size and translation precede GPA/shape PCA. |

These arbitrary-unit measurements are controlled numerical stress tests. Genomic
Binomial(2,p) assumes Hardy-Weinberg sampling conditional on the latent vector;
mixing over target/source effects can create overdispersion across observations.

The bundled eight scenarios are prespecified:

| Scenario | Change |
| --- | --- |
| `null_matched` | Zero group separation; one reference group. |
| `strong_matched` | All views carry both axes, strength 3. |
| `weak_matched` | All views carry both axes, strength 0.6. |
| `complementary_matched` | Genomic/ecological views carry only axis 1; traits/shape carry only axis 2. |
| `conflicting_matched` | Ecology uses a separately permuted balanced four-group partition. Other views retain the primary partition. |
| `missing_cells` | Strong matched structure plus independent 15% genomic/tabular cell missingness. Shape coordinates remain complete. |
| `missing_views` | Strong matched structure plus independent 25% removal of whole ecological records. |
| `unequal_grouped` | Complementary structure, explicit locality aggregation, and independently sampled 1–4 source individuals per locality in each specimen-level view. |

Matched datasets have one biological individual shared by the four views.
Grouped datasets use different source individuals for genomic, trait and shape
data, with explicit locality links and PC-score means; ecology is native to
locality level. Equal grouped sampling with two individuals per view is also
configurable. No ecological values are broadcast to individuals.

## Comparisons and truth isolation

The default arms are all four views, genomic plus ecology, and traits plus shape.
They use the same integrative estimator, joint dimension and declared settings.
They are view-subset comparisons, not single-modality fits or comparisons against
external software. Other subsets of two to four kinds can be declared.
The first listed arm is the reference for paired differences.

Before fitting, intersect coverage of **all four generated views**, using IDs
only. Stage that ordered common cohort for every arm, including arms that do
not use ecology. SNPio eligibility is then determined on that staged cohort.
`observed/` preserves the original unequal coverage; `cohort/` contains the
prespecified complete-case study inputs. This benchmark evaluates a fixed shared
cohort; it does not compare each arm's maximal available-data policy or correct
complete-case selection bias. Truth counts before/after selection remain visible.

Generating labels live in a separate `truth.json` beside the input directories.
They never appear in data configurations, feature tables or fit artifacts.
Only the scorer receives labels, after fitting finishes. Weights, dimensions,
GMM settings, arm choices and the 0.9 high-stability diagnostic threshold are
prespecified; no selection is optimized using benchmark results.

SHA-256-derived seeds depend on master seed, scenario name, repetition and
purpose. Data and analysis streams are distinct; existing estimator SeedSequence
children separate point-fit, bootstrap-draw and bootstrap-fit seeds. All arms
share analysis seeds and the same target order, producing paired target draws.
Adding/reordering arms, scenarios or repetitions leaves existing stream keys
unchanged. Renaming a scenario intentionally changes its streams. Reproducibility
is within recorded library/platform versions, not guaranteed bitwise across them.

## Performance measures and denominators

For each completed point fit, score the retained cohort using:

- Adjusted Rand index (ARI), selected K, exact K recovery and absolute K error.
  ARI is label-invariant and chance-adjusted; see the
  [scikit-learn definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html).
  Null ARI can equal one for two all-one-group partitions; also inspect false
  splitting (`selected K>1`). This is a clustering diagnostic, not a test's type-I error.
- Pair precision, recall and disagreement, using each unordered off-diagonal
  pair once. Zero-denominator ratios are missing, not zero. K truth refers to
  groups present in the retained cohort; generating and retained group counts
  expose attrition. Conflicting scenarios also report ARI against ecology's
  alternative partition; neither reference is asserted to be a biological species truth.
- Mean `(bootstrap_coassignment_ij - true_same_ij)²`, plus separate within-group
  and between-group mean frequencies. Report all-prediction and OOB versions.
  These are descriptive recovery errors, not proper species-probability scores.
- Exact truth-set matching for each point cluster and its exact bootstrap recovery.
  Report the fraction of clusters with frequency ≥0.9 that match a complete
  reference group, plus raw stable/wrong counts. Pure fragments do not count as
  correct groups. With no qualifying cluster this fraction is missing.

OOB error uses pairs with finite, unaffected OOB estimates only. Pair coverage
and the fraction ever requested OOB are explicit: ten draws leave many pairs
unobserved. OOB errors therefore describe a selected pair set, not all pairs.
Per-run ten-bin tables compare support and observed same-group fractions, with
pair counts. They are descriptive, without independent-pair error bars.

Any failed bootstrap withholds main coassignment point estimates. The scorer
propagates the existing failed-outcome frequency bounds into conservative
squared-error bounds: for truth 0 use `[low²,high²]`; for truth 1 use
`[(1-high)²,(1-low)²]`. Average over the stated pair set. These are missing-outcome
identification bounds, not confidence intervals. OOB bounds exclude pairs with
no requested OOB draws. Within-target members and source genomic eligibility
remain fixed in this bootstrap.

Every requested dataset/arm gets a ledger entry. Integrative workflow
ValueError/RuntimeError and numerical failures are recorded with their stage/type
and the study continues; I/O, scoring and cohort-contract
errors abort the study with `failure.json`. A completed study can contain model
failures. Inspect `point_failed`, `bootstrap_incomplete_runs`, and each group's
requested/successful counts. Requested inner draws include failed point fits;
they are never silently removed. There is no retry or automatic parameter repair.

Within each scenario/arm, report equal-dataset mean, median, SD, range and
`MCSE = sample_SD/sqrt(m)` for m available independent datasets. MCSE is missing
when m<2. Report `n_available`/`n_missing` against all requested datasets for every
metric, including completely failed arms. Availability can be selective, so these
means condition on availability. Paired differences are reference minus comparator
on repetitions with both metrics available, with paired MCSE and missing counts.
Scenarios are not pooled into a single accuracy estimate.

Pairs, clusters and inner draws are not independent simulation replicates.
Small m can give a zero estimated MCSE even when true simulation uncertainty is
substantial. The output also reports the worst-case Bernoulli rate MCSE
`0.5/sqrt(n_requested)`; the three-dataset smoke value is about 0.289.
No species p-values, confidence intervals, significance claims or formal support
calibration are supplied by this checkpoint.

## Configuration, artifacts and API

`examples/benchmarking/smoke.json` requests 8 scenarios × 3 datasets × 3 arms,
64 targets, 64 loci, 10 inner draws and 3 GMM starts. All view weights are equal;
two PCs per view and two joint PCs are fixed. Mean imputation is explicit for
genomic/tabular cells. Fixed landmarks are complete. The template requires all
four kind-named views at specimen level without aggregation and seed 0; the
runner sets actual levels/aggregation and derived seeds for each run.

Study JSON rejects unknown fields, duplicate/unsafe names, invalid numeric types,
unsupported patterns and incompatible templates. `n_units` is at least 20 and
divisible by four; loci are even and at least eight. Model numerical and pairwise
budgets also guard conservative simulation allocation. Disk use, run records and
elapsed time grow with study size and are not bounded by the per-fit memory limit.

`examples/benchmarking/study.json` specifies 100 datasets per scenario and 100 inner
draws (2,400 planned point fits and 240,000 planned inner refits). The subsequent
[larger-study checkpoint](integrative-study.md) executed this unchanged declaration
and retained all failures. Its worst-case rate MCSE is 0.05; study size should follow a chosen
precision target. More inner draws reduce OOB sparsity but cannot replace more
independent outer datasets. Edit the configuration before running a larger study:

```bash
super-deli benchmark-integrative examples/benchmarking/study.json \
  --output-dir /path/to/new-integrative-study
```

| Artifact | Contents |
| --- | --- |
| `config.json` | Exact validated study declaration |
| `scenario_NNNN/simulation/{observed,cohort}/` | Separate typed source files and dataset declarations |
| `simulation/truth.json`, `manifest.json` | Generating labels, cohort/coverage, source counts and input hashes |
| `scenario_NNNN/arm/` | Existing integrative numerical/provenance artifacts; no truth predictor |
| `arm/evaluation.json` | ID-aligned metrics, support bins, exact truth-set checks and denominators |
| `runs.jsonl` | Flushed per-attempt records, seeds, paths, outcomes and elapsed seconds |
| `metrics.csv` | Scenario/arm metrics with availability and MCSE |
| `summary.json` | Completion/failure counts, versions, hashes, metric summaries and paired differences |

Elapsed seconds include workflow ingestion, fitting and scoring on one host;
they are not isolated algorithm timings. Output paths are exclusive. The core
runner is sequential; the subsequent study checkpoint adds an external scenario
orchestrator that preserves repetition indices/seed keys. There is no resume.

```python
from super_deli.benchmarking.config import BenchmarkConfig
from super_deli.benchmarking.runner import run_benchmark
from super_deli.io.config import read_json

summary = run_benchmark(
    BenchmarkConfig.from_dict(read_json("examples/benchmarking/smoke.json")),
    output_dir="/tmp/super-deli-benchmark-api",
)
```

Config, simulation, scoring and orchestration are separate modules. Numerical
modules load only when requested; the foundation/config API remains dependency
free. Tests include hand-computed losses/MCSE, stable wrong partitions, null and
complementary generators, matched/grouped links, missing data, source preservation,
real SNPio, shared cohort/draw audits, reproducibility and injected failures.
