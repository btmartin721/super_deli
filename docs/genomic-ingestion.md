# SNPio genomic ingestion checkpoint

Python 3.11 and 3.12 remain the supported runtimes. The current SNPio release
declares Python `>=3.11,<3.13`; no Python 3.13/3.14 metadata or CI changes were made.
This checkpoint uses `snpio>=1.7.7,<2` and imports it only when ingestion is requested.

## API and command

```bash
python -m pip install -e '.[dev,genomics]'
super-deli ingest-genomic examples/genomic/dataset.json --output-dir /tmp/new-genomic-run
```

The parent directory must exist and the run directory must be new. It is never
reused or overwritten. If a configuration contains multiple genomic modalities,
select one with `--modality NAME`. Only that modality's contents are loaded;
all configuration declarations are still checked for consistency.

```python
from pathlib import Path

from super_deli import load_config
from super_deli.io.genomic import ingest_genomic

source = Path("examples/genomic/dataset.json").absolute()
loaded = ingest_genomic(
    load_config(source),
    base_dir=source.parent,
    output_dir=Path("/tmp/new-genomic-api-run"),
    modality_name="genotypes",
)
genotype_data = loaded.genotype_data
sample_ids = loaded.sample_ids
specimen_ids = loaded.specimen_ids
missing_mask = loaded.missing_mask
```

The returned object contains the SNPio `GenotypeData` reader object. Retain the
output directory while using its VCF metadata and caches. The SNPio object is
mutable; the returned sample/specimen tuples and read-only missing-call mask
describe ingestion time. No global working-directory or environment changes are
made by the API. The CLI redirects SNPio console output to stderr so stdout
contains one JSON summary.

## Identity and population rules

Each genomic observation ID must match one source sample ID exactly. Case,
spelling, missing IDs, duplicate IDs, and unexpected samples are checked. The
configuration observation order may differ from the source order. The adapter
returns specimen links in source sample order and never joins by row position.

Multiple source records may map to one specimen, preserving technical replicates.
They remain separate matrix rows; they are not aggregated or treated as independent
biological individuals. Summaries give both sample and unique-specimen counts.

External popmaps are optional. Provided maps must contain every selected source
sample exactly once, with no additional samples. Two whitespace/tab-separated
columns or CSV are accepted, with an optional SNPio-compatible sample/population
header. Empty, padded, duplicate, or incomplete mappings are rejected before
SNPio can convert them to a dictionary or subset the alignment.

SNPio receives `popmapfile=None` when no external file is supplied. Embedded
STRUCTURE population IDs are retained when declared with `has_popids=true`.
GENEPOP population sections are retained as SNPio's `Pop1`, `Pop2`, etc. These
are sampling groups, not inferred species. For SNPio 1.7.7, a small GenePopReader
subclass preserves the labels that its parser has already read before the base
class replaces absent-map metadata with `NA`. It uses public population setters
and performs no genotype parsing of its own.

When explicit specimen population metadata is present, it must agree with the
provided/embedded population label for that source sample. When no population
information is supplied, the summary records source `none` and null population
labels; SNPio's internal `NA` fallback is not interpreted as a biological group.
Source population labels are not automatically added to the configuration's
population registry or used to link other modalities.

## Reader options

The optional `reader_options` object accepts only these fields:

| Field | Formats | Meaning |
| --- | --- | --- |
| `has_popids` | STRUCTURE | Population IDs occur after sample IDs; default false |
| `has_marker_names` | STRUCTURE | Marker names are in the first line; default false |
| `allele_start_col` | STRUCTURE | Zero-based first allele column; default SNPio layout |
| `allele_encoding` | STRUCTURE, GENEPOP | Explicit string allele-code to A/C/G/T/N map |

STRUCTURE one-row and two-row layouts are detected by SNPio. Custom encoding
maps must include every token required by the reader, including `"-9": "N"`
for missing STRUCTURE calls when that token is used. GENEPOP codes such as `01`
must remain strings so leading zeros are preserved. Encoding-map order is
irrelevant and canonicalized; ordinal trait-category order remains meaningful.
Arbitrary reader kwargs, population filters, and imputation options are not accepted.

## Outputs and provenance

The run directory contains staged input/popmap copies, SNPio-generated artifacts
under `snpio_output/`, `genotypes.npy`, `missing_mask.npy`, and `summary.json`.
Arrays are written without pickle and contain parsed IUPAC calls and a Boolean
missing-call mask. Missing calls are SNPio's N, dash, dot, or question-mark codes;
IUPAC heterozygotes such as M or R are observed calls.

Original files are copied, and source/snapshot SHA-256 fingerprints must agree
before parsing. SNPio may then create compressed VCFs, indexes, logs, HDF5 caches,
or derived popmaps inside the run directory. No source file is modified. Staging
and fingerprinting perform additional full-file I/O in bounded memory and require
disk space for copies and SNPio artifacts. This operation is explicit; ordinary
declaration validation does not stage or load data.

The summary records source/sample order, specimen links, dimensions, missingness,
population metadata, the configuration snapshot/digest, runtime/software versions,
and selected-source hashes. Hash status `computed_selected_modality` means only
the selected genomic source and optional external popmap were hashed. Other
modalities are not implicitly audited. Preserve the complete output directory;
the NumPy arrays alone do not replace SNPio's original allele/VCF metadata.

Errors before run creation leave no run directory. Errors after creation retain
`failure.json` and staged diagnostics, without a successful `summary.json`.
Start a new directory after correcting the cause; this checkpoint does not resume
or overwrite failed runs. Ingestion errors use CLI exit code 2. Declaration-only
validation retains its existing 0/1/2 exit-code contract.

SNPio logger setup is process-global. This adapter serializes its own ingestion
calls, temporarily detaches caller SNPio handlers, closes only handlers created
during ingestion, and restores the caller's handlers/levels. Other code using
SNPio concurrently should run in a separate process. A returned object's later
SNPio operations follow the caller's logger configuration.

## Validation boundary

The tiny, hand-authored examples specify two samples, four loci, one missing
call, and known IUPAC call truth across formats. Tests cover external/no popmaps,
STRUCTURE layout variants, custom encodings, sample-order changes, repeated
specimens, metadata conflicts, source preservation, and output non-overwrite.
These fixtures validate parsing and identity linkage. They are not empirical
benchmarks or simulated species truth.

Parsed calls retain SNPio's format-specific normalization behavior; raw sequence
strings and multiallelic information should not be reconstructed from the NumPy
matrix alone. The adapter adds no genotype filtering, imputation, feature scaling,
clustering, species assignments, or statistical uncertainty. Empirical datasets,
scientific QC, estimator design, and calibration remain later checkpoints.
