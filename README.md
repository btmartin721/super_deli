# Super Deli

Super Deli is being redesigned as a modular Python package for integrative
species-delimitation research using genomic, morphological, ecological, and
geometric morphometric data.

**Current checkpoint: bounded grouped AIC/likelihood power and recovery validation.**
See the [frozen method](docs/grouped-power-recovery.md) and
[completed results](docs/grouped-power-recovery-results.md).
All 400 datasets and 40,780 planned fits completed without failures.
At the original signal strength, likelihood detection was 48% with 64 localities
and 100% with 256; AIC selected four components in 26% and 96%, respectively.
AIC also selected multiple components in 18/25 Gaussian controls at N=64, so
component count alone is unsuitable as species evidence. BIC defaults and
historical p-values remain unchanged; AIC remains opt-in.

The preceding [larger false-positive validation](docs/grouped-false-positive-validation-results.md)
completed 2,500 datasets and 497,500 fits; fitted-null rejection was 4.4–5.4%.
The present small control cells do not replace that study. General calibration
and species-boundary validity remain unestablished.
Genomic, numeric morphology/ecology and shape scores can be fused by exact unit
IDs or explicit population/locality aggregation, with prespecified view weights.
The package also supports individual-modality PCA/GMM, frozen-fit shape
projection, visible marker detection and multimodal ingestion.
Anatomical landmark learning and formal species-boundary tests remain later work.

The new `benchmark-integrative` command evaluates the existing integrative model
and declared view subsets against separate generating partitions. It includes
eight stress scenarios, matched/grouped sampling, explicit failure denominators,
and independent-dataset Monte Carlo standard errors. See the
[benchmark protocol](docs/integrative-benchmarking.md) and
[bounded smoke results](docs/integrative-benchmark-smoke.md).
The [larger study results](docs/integrative-study.md) now report 800 simulated
datasets, 2,400 point-fit attempts, explicit failures and audited bootstrap logs.
`analyze-integrative-study` audits completed scenario shards and exports compact
dataset, paired-comparison, event-rate and support-bin tables without refitting.

The separate `calibrate-integrative` command tests a single Gaussian against the
configured GMM search in frozen joint scores, using an approximate parametric
bootstrap. It preserves the original clustering run and makes no species-boundary
or cluster-specific significance claim. See the [method and validation protocol](docs/gaussian-score-calibration.md).

The [completed calibration study](docs/gaussian-score-calibration-results.md)
reports 400 independent score datasets with explicit null, power and stress interpretations.

The separate `calibrate-reference-pipeline` command regenerates all four raw
modalities and refits ingestion, preprocessing, GPA, PCA, aggregation, fusion
and the complete GMM search for each Monte Carlo draw. Its global test uses a
known continuous-population simulator with explicit assumptions. See the
[reference-law method](docs/full-pipeline-calibration.md) and
[100-dataset validation results](docs/full-pipeline-calibration-results.md).
The empirical extensions below cover specific declared nulls; general empirical
nulls and species-boundary significance remain future work.

The new `calibrate-empirical-integrative` command estimates a joint Gaussian
null from matched, complete continuous morphology/ecology before PCA and refits
the integrative pipeline for every bootstrap draw. Its p-values are approximate.
See the [method](docs/empirical-null-calibration.md) and
[validation results](docs/empirical-null-calibration-results.md).
`calibrate-grouped-integrative` additionally fits joint between-group and
within-morphology Gaussian covariance with unequal specimen counts and one ecology
row per population/locality. It preserves the observed sampling design through
full-pipeline bootstrap refits, rejects unsupported covariance fits and keeps
missing-test denominators explicit. See the [grouped method](docs/grouped-null-calibration.md) and
[100-dataset results with explicit withheld tests](docs/grouped-null-calibration-results.md).
Schema-2 grouped calibration adds positive-semidefinite between-group covariance,
bounded convergence checks and a versioned sampler while retaining legacy audits.
See the [boundary method](docs/grouped-covariance-boundaries.md) and
[175-dataset results](docs/grouped-covariance-boundaries-results.md).
Additional fitted nulls for genomic/count/shape data remain separate checkpoints.

## Installation

Use Python 3.11 or 3.12. From this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The foundation has no runtime dependencies. Install the genomic extra to enable
SNPio ingestion on Python 3.11 or 3.12:

```bash
python -m pip install -e '.[dev,genomics]'
```

## Validate the illustrative dataset

```bash
super-deli validate examples/foundation/dataset.json --hash-inputs
```

The example includes specimen measurements, repeated observations, partial
specimen coverage, and locality-level environmental observations. Values are
illustrative and are not an empirical species-delimitation benchmark.

The command prints a JSON validation report, coverage counts, and provenance.
Add `--report /path/to/new-report.json` to also save it. Existing files are never
overwritten. Relative source references resolve against the configuration's
directory, independently of the working directory. `python -m super_deli`
provides the same CLI.

Exit codes: `0` for valid declarations/references, `1` for consistency/reference
errors, and `2` for malformed configuration, I/O, or usage errors. A successful
report still warns that modality contents and model suitability are unchecked.

## Python API

```python
from pathlib import Path

from super_deli import load_config, validate_dataset

source = Path("examples/foundation/dataset.json").absolute()
config = load_config(source)
report = validate_dataset(config, base_dir=source.parent)
print(report.to_dict())
```

See [the architecture and configuration contract](docs/architecture.md) for
biological levels, feature types, optional popmaps, uncertainty reporting rules,
validation boundaries, and the next checkpoints.

## Load genomic data

```bash
super-deli ingest-genomic examples/genomic/dataset.json --output-dir /tmp/super-deli-genomic-example
```

Choose a new output directory for each invocation. The adapter accepts VCF,
PHYLIP, STRUCTURE, and GENEPOP through SNPio's readers, with `popmapfile=null`
when no external map is supplied. It stages private input copies, preserves
source sample order, checks sample/specimen links, and records parsed IUPAC
calls, their missing-call mask, and provenance. Original inputs are unchanged.

```python
from super_deli.io.genomic import ingest_genomic

source = Path("examples/genomic/dataset.json").absolute()
loaded = ingest_genomic(
    load_config(source),
    base_dir=source.parent,
    output_dir=Path("/tmp/super-deli-genomic-api-example"),
)
genotype_data = loaded.genotype_data  # SNPio GenotypeData object
```

See [genomic ingestion](docs/genomic-ingestion.md) for reader options, outputs,
population metadata, and the precise validation boundaries. The hand-authored
fixtures specify expected genotype calls; they are not species-delimitation
benchmarks.

## Estimate candidate genetic clusters

```bash
python -m pip install -e '.[dev,genomics,baseline]'
super-deli baseline-genomic examples/baseline/dataset.json \
  --model-config examples/baseline/model.json \
  --blocks examples/baseline/blocks.json \
  --output-dir /tmp/super-deli-baseline-example
```

The baseline uses diploid biallelic dosages, explicit missingness/scaling policies,
PCA, and BIC selection of diagonal Gaussian mixtures including K=1. Bootstrap
replicates refit the complete pipeline using declared independent blocks or an
explicit unlinked-locus assumption. Conditional memberships, pairwise
co-assignment, and exact-cluster recovery remain distinct. Failed replicates are
retained; incomplete bootstrap estimates are withheld and bounded.

See [the baseline method and API](docs/genomic-baseline.md) for assumptions,
configuration, artifacts, reproducible synthetic truth, and validation limits.
The synthetic example is a software check, not empirical species validation.

## Load morphology and ecology

```bash
super-deli ingest-tabular examples/tabular/dataset.json \
  --output-dir /tmp/super-deli-tabular-example
```

The base installation supports continuous measurements, counts, binary states,
ordered scores and categorical traits, including units and covariates. Source
observation IDs must match declarations; repeated measurements and specimen,
population or locality levels remain explicit. The reader preserves missingness
and category labels. Select tables with repeated `--modality NAME` options.

See [tabular ingestion](docs/tabular-ingestion.md) for column/missing-token
options, the Python API, typed JSON artifacts, validation and processing limits.
The hand-authored example provides value/link truth for ingestion checks.

## Load geometric morphometric coordinates

```bash
super-deli ingest-morphometric examples/morphometric/dataset.json \
  --output-dir /tmp/super-deli-landmark-example
```

The base installation accepts fixed-landmark TPS (`LM=`/`LM3=`) and long-form
coordinate CSV. Declare dimensions, landmark order, biological observation links
and missing-value rules. Readers preserve coordinates, partial missingness,
repeat observations and scale/image metadata. Scale factors are recorded without
rescaling. This coordinate adapter retains images as metadata references.

See [morphometric ingestion](docs/morphometric-ingestion.md) for the Python API,
supported format subset, coordinate correspondence, artifacts and limitations.

## Load images

```bash
python -m pip install -e '.[dev,images]'
super-deli ingest-images examples/images/dataset.json \
  --output-dir /tmp/super-deli-image-example
```

Image manifests link each selected page/frame to a declared observation and
biological unit. The reader keeps verified source copies, writes RGB/RGBA PNG
display derivatives, and records orientation, resolution, bit depth and hashes.
Multipage/multiframe inputs require explicit selection. Pixel/file-size budgets
and explicit HEIF bit-depth policies prevent silent resizing or quantization.

See [image ingestion](docs/image-ingestion.md) for the manifest/options contract,
Python API, source preservation, color/coordinate limits and synthetic codec truth.

## Detect visible landmark markers

```bash
python -m pip install -e '.[dev,images,landmarks]'
super-deli detect-landmarks examples/detection/dataset.json \
  --detector-config examples/detection/detector.json \
  --output-dir /tmp/super-deli-detected-landmarks
```

Declare named RGB/search-region rules and component-size limits. Detection leaves
missing, ambiguous and potentially clipped markers null. Separate overlays and
diagnostics support review; coordinate JSON/CSV and an ingestible configuration
preserve biological links. Every centroid remains unreviewed, and statistical
localization uncertainty is explicitly unavailable pending calibration.

See [landmark detection](docs/landmark-detection.md) for configuration, the API,
coordinate conventions, synthetic accuracy checks and method boundaries.

## Align landmarks and estimate shape axes

```bash
python -m pip install -e '.[dev,shape]'
super-deli shape-pca examples/shape/train_dataset.json \
  --model-config examples/shape/model.json --output-dir /tmp/super-deli-shape-fit
super-deli shape-pca examples/shape/projection_dataset.json \
  --model-config examples/shape/model.json --reference-run /tmp/super-deli-shape-fit \
  --output-dir /tmp/super-deli-shape-projection
```

GPA removes translation, uniform size and rotation from homologous 2D/3D fixed
landmarks. Reflections are forbidden by default. Missing landmarks and repeated
units require explicit policies. Tangent-space covariance PCA records numerical
rank, explained variance and geometric distortion; saved parameters project new
observations without updating the training fit. This is descriptive ordination.

See [shape analysis](docs/shape-pca.md) for equations, policies, artifacts, API,
analytic truth tests and the provisional check of the supplied TPS download.

## Cluster shape scores and assess stability

```bash
python -m pip install -e '.[dev,clustering]'
super-deli cluster-shape examples/shape/clustering/dataset.json \
  --model-config examples/shape/clustering/model.json \
  --output-dir /tmp/super-deli-shape-clusters
```

This fits candidate shape groups using BIC (including K=1) in a fixed PC
dimension. Bootstrap draws resample biological units and refit GPA, PCA and
mixture selection. Repeated digitizations require aggregation first. Conditional
membership, bootstrap frequency and failed-replicate bounds are separate;
species-boundary p-values remain uncomputed. See
[shape clustering](docs/shape-clustering.md) for assumptions, OOB denominators,
configuration, artifacts, API and bounded synthetic validation.

## Cluster numeric morphology or ecology

```bash
python -m pip install -e '.[dev,clustering]'
super-deli cluster-tabular examples/tabular/clustering/dataset.json \
  --model-config examples/tabular/clustering/morphology_model.json \
  --modality traits --output-dir /tmp/super-deli-morphology-clusters
super-deli cluster-tabular examples/tabular/clustering/dataset.json \
  --model-config examples/tabular/clustering/ecology_model.json \
  --modality environment --output-dir /tmp/super-deli-ecology-clusters
```

This baseline analyzes one table of continuous/count features per run, with
explicit selection, scaling, count transformation and missingness policies.
Whole-unit bootstrap draws refit preprocessing, PCA and BIC/GMM selection.
Outputs preserve specimen, population or locality levels. Categories require
a numeric subset; repeated units require aggregation before fitting. See
[tabular clustering](docs/tabular-clustering.md) for assumptions, API, artifacts
and limits on interpreting environmental groups as biological species.

## Integrate linked modalities

```bash
python -m pip install -e '.[dev,clustering,genomics]'
super-deli cluster-integrative examples/integrative/grouped_dataset.json \
  --model-config examples/integrative/grouped_model.json \
  --output-dir /tmp/super-deli-integrative-groups
super-deli cluster-integrative examples/integrative/matched_dataset.json \
  --model-config examples/integrative/matched_model.json \
  --output-dir /tmp/super-deli-integrative-individuals
```

Select two to four view kinds and an explicit target biological level. The
baseline normalizes modality score profiles, fits joint PCA/GMM, and resamples
target units jointly across all views. Unmatched individuals require explicit
group links; entire missing views require an explicit complete-case policy.
Source memberships, exclusions, weights and numerical parameters are exported.
See [integrative clustering](docs/integrative-clustering.md) for the method,
configuration, reconstruction and conditional uncertainty scope. Components
remain candidate structure; formal species significance is uncomputed.

## Development checks

```bash
python -m pytest -q
ruff check
ruff format --check
python -m build
```

The test suite includes the read-only image/TPS manifest utility.
GitHub Actions tests the foundation on Python 3.11 and 3.12, validates the example,
builds wheel/source distributions, and checks wheel installation outside the
source checkout, including tabular and coordinate ingestion without scientific dependencies.
A separate genomic matrix tests the real SNPio readers with
and without external popmaps, the numerical baseline, real image codecs, visible
marker detection, GPA/shape PCA, unimodal/integrative clustering, and installed CLIs.
Optional tests are skipped
when extras are absent. Ruff coverage is declared centrally in `pyproject.toml`.
Linting covers the new package, tests, and synthetic-example generator.

The active implementation lives under `src/super_deli/`. Obsolete imputation
code and personal datasets are archived outside the repository.
The standalone `image_tps_manifest.py` keeps its existing manifest schema and
metadata checks, with PDF/HEIC/HEIF extensions added to discovery.

The [genomic fitted-null checkpoint](docs/genomic-null-calibration.md) adds an
approximate full-pipeline pooled-HWE test for explicitly unlinked diploid SNPs,
with optional popmaps and preserved failed-draw accounting.
See the [completed genomic validation report](docs/genomic-null-calibration-results.md)
for the frozen 125-dataset study and its inferential limits.

The [linkage-aware genomic checkpoint](docs/linked-genomic-calibration.md) adds
a separate fitted empirical block-profile null for complete genotypes and
prespecified independent blocks.
See the [completed linkage-aware validation](docs/linked-genomic-calibration-results.md)
for the frozen study, paired outcomes and sparse-support limitations.

The [joint genomic–phenotypic fitted-null workflow](docs/joint-genomic-phenotypic-calibration.md)
combines empirical genotype-block profiles with conditional Gaussian regression
of matched continuous morphology/ecology on prespecified raw genomic summaries.
It reruns the complete integrative pipeline and reports approximate joint-law
departure p-values, with stability and species-boundary interpretation separate.
See the [completed 125-dataset validation](docs/joint-genomic-phenotypic-calibration-results.md) for outcomes and limitations.

The [four-group recovery diagnostics](docs/integrative-recovery-diagnostics.md)
audit the generating signal, PCA retention, mixture constraints, optimization
and BIC penalties in the constructed joint-study alternative. They preserve
all models and significance results; the frozen grid is diagnostic only.
See the [completed recovery report](docs/integrative-recovery-diagnostics-results.md)
for all 525 point fits, the separate optimization diagnostic and its limitations.

An opt-in [shared-covariance integrative model](docs/shared-covariance-clustering.md)
reduces mixture complexity while retaining correlated joint scores. The
full-covariance default and existing significance results are preserved.
See [repository layout and local data](docs/repository-layout.md) for the cleaned
source tree and external-data policy.
See the [completed paired validation](docs/shared-covariance-clustering-results.md)
for recovery gains, unequal-variance/cline limitations and audited bootstrap support.

Shared-covariance full-pipeline significance calibration is available for the
matched genomic–phenotypic empirical null. See the
[scope and frozen validation](docs/shared-covariance-calibration.md). The covariance
family must be fixed before analysis; the default remains full covariance.

The [completed shared-family calibration](docs/shared-covariance-calibration-results.md)
reports all 125 fresh tests, conditional Monte Carlo intervals and stress limitations.

Shared-covariance [grouped morphology–ecology calibration](docs/shared-grouped-calibration.md)
uses the boundary-capable schema-2 raw null, explicit group links and fixed unequal
specimen counts. Legacy schema 1 retains its full-family scope. The covariance
family must be selected before analysis; stability remains separate.

The [completed grouped shared-family study](docs/shared-grouped-calibration-results.md)
reports 175 fresh observations, full-pipeline calibration and uncertainty.

The frozen [grouped recovery diagnostic](docs/grouped-recovery-diagnostics.md)
traces morphology/ecology signal through source PCA, group aggregation, fusion
and joint PCA. The [completed diagnosis](docs/grouped-recovery-diagnostics-results.md)
reports 750 point attempts (700 complete, 50 retained numerical failures), 75 separate
truth-initialized diagnostics and 25 reproduced archived baselines. Defaults and
existing p-values remain unchanged; exploratory settings are not promoted.

An opt-in [group-mean morphology PCA policy](docs/grouped-pca.md) gives each linked
population/locality equal influence on preprocessing and PCA. It supports frozen
projection and whole-group stability. Significance is now available within the
[bounded schema-2 grouped workflow](docs/groupmean-calibration.md). The [completed four-arm study](docs/grouped-pca-results.md) reports all
700 point workflows and 1,000 stability refits. Four joint PCs improved recovery
at N=256; group-mean PCA added no consistent benefit in these laws. Defaults and
historical p-values remain unchanged.

[Group-mean PCA significance calibration](docs/groupmean-calibration.md) supports
complete continuous grouped morphology and ecology using the schema-2 raw
hierarchical Gaussian null and a prespecified shared-covariance GMM. Every raw draw
refits group means, preprocessing/PCA, integration and the full K search. This
opt-in approximate departure test preserves defaults and historical results.

The [completed group-mean calibration study](docs/groupmean-calibration-results.md)
reports all 175 observations, full-pipeline checks and uncertainty.

[Full-covariance group-mean PCA calibration](docs/full-groupmean-calibration.md)
uses the same schema-2 grouped raw null with a prespecified covariance per mixture
component. Both covariance families are supported; fix one before analysis.
Defaults and historical p-values are preserved.

The [completed full-covariance group-mean study](docs/full-groupmean-calibration-results.md)
reports all 175 observations, full-pipeline checks and uncertainty.


The [Gaussian-control AIC diagnostic](docs/aic-gaussian-diagnostics.md) examines
finite-sample component selection, unequal-count measurement variance, optimization
and independent prediction. It preserves production fitting, defaults and all
historical p-values; detailed evidence stays local.

The [completed Gaussian-control diagnosis](docs/aic-gaussian-diagnostics-results.md)
reports 1,250 verified searches and the audit of 2,550 archived observations,
with uncertainty and independent held-out prediction.
