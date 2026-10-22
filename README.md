# Super Deli

Super Deli is being redesigned as a modular Python package for integrative
species-delimitation research using genomic, morphological, ecological, and
geometric morphometric data.

**Current checkpoint: genomic ingestion.** The package validates dataset
declarations and loads genomic inputs through SNPio. It does not fit models,
delimit species, compute uncertainty, or decode images yet. Other modalities
remain declarations for later ingestion checkpoints.

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

## Development checks

```bash
python -m pytest -q
ruff check src tests/test_foundation*.py tests/test_genomic*.py
ruff format --check src tests/test_foundation*.py tests/test_genomic*.py
python -m build
```

The test suite includes the unchanged read-only image/TPS manifest utility.
GitHub Actions tests the foundation on Python 3.11 and 3.12, validates the example,
builds wheel/source distributions, and checks wheel installation outside the
source checkout. A separate genomic matrix tests the real SNPio readers with
and without external popmaps. Integration tests are skipped when the genomic
extra is absent. Linting covers the new package and its tests.

Historical imputation sources remain outside the installed `src/` package.
Their previous documentation is preserved in [the legacy README](docs/legacy-readme.md).
The existing `image_tps_manifest.py` source utility remains available; its manifest
format will be reused in a later morphometrics-ingestion checkpoint.
