# Foundation checkpoint

Super Deli is being redesigned for integrative species-delimitation research.
The foundation provides an installable package, data declarations, validation,
provenance, and result conventions. The second checkpoint adds SNPio genomic
ingestion. Neither checkpoint contains delimitation estimators.

## Boundaries

| Module | Responsibility |
| --- | --- |
| `core.data` | Immutable specimen, observation, feature, and dataset declarations |
| `core.schema` | Structural parsing and configuration version checks |
| `core.validation` | Biological references, feature declarations, and file availability |
| `core.provenance` | Configuration snapshots, software versions, and optional input hashes |
| `core.results` | Assignment and uncertainty reporting contracts for future estimators |
| `io.config` | Strict JSON reading and non-overwriting artifact writing |
| `io.genomic` | Optional SNPio readers, staged inputs, sample linkage, and ingestion artifacts |
| `cli` | Argument handling and presentation of API results |

The core does not import scientific libraries or perform modality I/O. Future
readers belong in `io`; estimators, statistical evaluation, and presentation will
be added when those features have tested implementations. Empty plugin systems
and unused model classes are intentionally absent from this checkpoint.

The distribution discovers only packages under `src/`. Historical imputation
code remains in the repository but is excluded from the installed package. The
previous README is retained in [legacy-readme.md](legacy-readme.md).

## Configuration version 1

Configurations are UTF-8 JSON. Duplicate keys, unknown fields, nonfinite JSON
constants, unsupported schema versions, and malformed field types are errors.
Identifiers are nonempty, case-sensitive strings without surrounding whitespace.
They are never automatically trimmed, uppercased, or matched by row position.

Top-level fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Required integer `1` |
| `dataset_id` | Required dataset identifier |
| `random_seed` | Nonnegative integer; defaults to `0` |
| `specimens` | Specimen objects with `specimen_id` and optional `population_id`/`locality_id` |
| `populations`, `localities` | Explicit arrays of identifiers used by biological links |
| `modalities` | Required, nonempty array of source declarations |

At least one biological unit and one modality must be declared. Specimens are
optional for a dataset containing only population- or locality-level data.
Population and locality identifiers are sampling metadata, never species labels.
Known taxonomic labels and benchmark truth should remain separate evaluation
inputs rather than features used to discover the same labels.

Each modality has `name`, `kind`, `level`, `format`, `path`, and `observations`.
An observation has an `observation_id` identifying its source record and a
`unit_id` linking it to a declared biological unit at that modality's `level`.
Names are unique across modalities; observation identifiers are unique within a
modality. Multiple observations can refer to the same biological unit.

| `kind` | Declared `format` values | Additional declarations |
| --- | --- | --- |
| `genomic` | `vcf`, `phylip`, `structure`, `genepop` | Specimen level; optional `popmapfile`, default `null` |
| `morphological` | `csv`, `tsv` | Nonempty analytical feature schema |
| `ecological` | `csv`, `tsv` | Nonempty analytical feature schema |
| `morphometric` | `tps`, `landmarks_csv`, `image_manifest` | Coordinate or manifest reference |

The `validate` command checks declarations and references only. The separate
`ingest-genomic` command loads genomic contents through SNPio into `GenotypeData`,
with `popmapfile=None` when absent. Other modality contents remain unchecked.
The optional `genomics` extra enables this dependency boundary on Python 3.11/3.12.
Genomic modalities may declare `reader_options` for STRUCTURE layout and
STRUCTURE/GENEPOP allele encodings; see [genomic ingestion](genomic-ingestion.md).

Tabular features declare `name`, `dtype`, optional `unit`, optional `categories`,
and `role` (`feature` or `covariate`, default `feature`). Types are `continuous`,
`count`, `binary`, `ordinal`, and `categorical`. Binary features require exactly
two states; ordinal categories require at least two states in their intended
order; unordered categorical features require at least one declared state.
Numeric features cannot declare categories. Units are recorded but not converted.
Covariates are recorded but no adjustment is performed.

## Linking partial and unpaired data

The example in `examples/foundation/` contains measurements of two out of three
specimens, a repeated measurement of one specimen, and ecological observations
at two localities. The report preserves all declared specimens and separately
counts observations, unique biological units, repeat observations, and missing
units within each modality's level.

Different individuals can carry different modalities and share explicit population
or locality links. These links do not establish specimen correspondence. Group
measurements are not broadcast to individuals or counted as independent specimen
replicates. Joint inference from unpaired data needs a model whose assumptions
make it identifiable; the foundation performs no automatic fusion or aggregation.

Models will need explicit capabilities for modalities, inference levels, and
missingness. Accepting a configuration does not establish that any particular
future model can analyze it.

## Images and existing TPS utility

`image_tps_manifest.py` and its existing tests remain unchanged. The source
distribution retains the script; the installed foundation package does not expose
it as a new reader. Its approved CSV columns remain:

`image_path`, `specimen_id`, `group`, `split`, `tps_path`, `landmark_count`,
`scale_value`, `scale_unit`, `check_status`, `check_details`.

An `image_manifest` source refers to that artifact. The new validator checks the
manifest file reference, not its rows or referenced images. Later ingestion must
make group/population associations explicit and honor the existing utility's
identifier checks. PDF page selection, PNG/JPEG and HEIC/HEIF decoding, orientation,
landmark localization, and coordinate transforms require later implementation and
validation. An image of a marker is not assumed to contain a TPS coordinate record.

## Provenance and output safety

Relative source paths resolve against the configuration file's parent directory;
absolute paths are accepted. Validation reads the configuration, checks declared
files are regular and readable, and writes nothing by default. With `--report`,
only a new report file is created; an existing path is never replaced. No output
directory is created automatically.

Reports embed the parsed configuration, its canonical-JSON SHA-256, random seed,
resolved base directory, timestamp, platform, Python version, and package version.
`--hash-inputs` hashes directly referenced data/popmap files in bounded memory.
Hashing is opt-in because genomic inputs can be large. A change detected during
hashing is an error. Image manifests are not recursively traversed. Hash status
explicitly distinguishes `computed`, `not_requested`, and `skipped_invalid_dataset`.
Hashes establish byte identity; they do not validate biological contents.

CLI exit codes are `0` for valid declarations/references, `1` for semantic or
reference errors, and `2` for malformed configurations, I/O errors, or usage errors.
Every completed validation report includes `contents_not_checked` as a warning.
Programmatic callers may use `check_files=False` for declaration-only checks;
the report records whether file checking was requested.

## Statistical reporting contract

`DelimitationResult` represents future candidate assignments at one declared
biological level. It requires unique unit assignments and explicit evidence.
The configuration SHA-256 links a result to its configuration/provenance record.
An unresolved assignment uses `label=None` and a reason. Each evidence record
declares its target, kind, method, and status. Unavailable or uncomputed evidence
requires an explanation and cannot contain a numeric estimate.

Bootstrap frequencies require a replicate count. Membership probabilities remain
distinct from bootstrap frequencies. Confidence intervals require finite, ordered
endpoints and a confidence level. P-values require a null hypothesis and an
explicit multiplicity treatment, including `none` if unadjusted. These checks
enforce reporting structure; they do not establish statistical validity.

Each estimator checkpoint must independently validate its assumptions, uncertainty
calibration, and interpretation. Evaluation must account for genomic linkage,
spatial dependence, repeated observations, model/cluster selection, and training
leakage. Candidate clusters must remain distinguishable from validated species
hypotheses. Simulated truth is conditional on the generating model; empirical
reference labels need independent evidence and documented uncertainty.

## Checkpoint sequence

1. **Completed foundation:** Package, contracts, validation, provenance, tests, CI, and a draft PR.
2. **Current:** SNPio ingestion and supported-format fixtures with optional popmaps.
3. A transparent baseline estimator and independently validated uncertainty.
4. Tabular morphology/ecology and landmark/image ingestion in separate batches.
5. Additional models and integrative inference after baseline benchmarks.

No models, image decoders, empirical-data downloads, or expensive analyses are
part of the genomic ingestion checkpoint.
