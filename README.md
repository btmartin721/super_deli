# Super Deli

Super Deli is being redesigned as a modular Python package for integrative
species-delimitation research using genomic, morphological, ecological, and
geometric morphometric data.

**Current checkpoint: integrative clustering with paired bootstrap stability.**
Genomic, numeric morphology/ecology and shape scores can be fused by exact unit
IDs or explicit population/locality aggregation, with prespecified view weights.
The package also supports individual-modality PCA/GMM, frozen-fit shape
projection, visible marker detection and multimodal ingestion.
Anatomical landmark learning and formal species-boundary tests remain later work.

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

Historical imputation sources remain outside the installed `src/` package.
Their previous documentation is preserved in [the legacy README](docs/legacy-readme.md).
The standalone `image_tps_manifest.py` keeps its existing manifest schema and
metadata checks, with PDF/HEIC/HEIF extensions added to discovery.
