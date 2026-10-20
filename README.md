# Super Deli

Super Deli is being redesigned as a modular Python package for integrative
species-delimitation research using genomic, morphological, ecological, and
geometric morphometric data.

**Current checkpoint: genomic PCA/GMM baseline.** The package validates dataset
declarations, loads genomic inputs through SNPio, and estimates candidate genetic
clusters with conditional memberships and block-bootstrap stability. These are
exploratory genetic clusters; species-boundary tests are not implemented. Other
modalities remain declarations for later ingestion checkpoints.

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

## Development checks

```bash
python -m pytest -q
ruff check src tests/test_foundation*.py tests/test_genomic*.py tests/test_baseline*.py examples/baseline/*.py
ruff format --check src tests/test_foundation*.py tests/test_genomic*.py tests/test_baseline*.py examples/baseline/*.py
python -m build
```

The test suite includes the unchanged read-only image/TPS manifest utility.
GitHub Actions tests the foundation on Python 3.11 and 3.12, validates the example,
builds wheel/source distributions, and checks wheel installation outside the
source checkout. A separate genomic matrix tests the real SNPio readers with
and without external popmaps, plus the numerical baseline and full analysis CLI.
Optional numerical/reader tests are skipped when their extras are absent.
Linting covers the new package, tests, and synthetic-example generator.

Historical imputation sources remain outside the installed `src/` package.
Their previous documentation is preserved in [the legacy README](docs/legacy-readme.md).
The existing `image_tps_manifest.py` source utility remains available; its manifest
format will be reused in a later morphometrics-ingestion checkpoint.
