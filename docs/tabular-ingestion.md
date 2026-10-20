# Morphology and ecology tables

`ingest-tabular` loads morphology and ecological measurements from CSV or TSV,
using the existing version-one feature schema and biological links. It uses only
the Python standard library on Python 3.11/3.12. No new models are added in this
checkpoint.

```bash
python -m pip install -e '.[dev]'
super-deli ingest-tabular examples/tabular/dataset.json \
  --output-dir /tmp/super-deli-tabular-example
```

The default is all declared `morphological` and `ecological` modalities. Select
one or more with repeated `--modality NAME` options. Tables are returned in
configuration order, independently of selection-argument order. Genomic and
morphometric sources are not opened; unselected files need not be available.
The complete dataset's declarations must still pass structural/link validation.
Relative source paths resolve against the configuration's directory. Use a new
output directory for each run.

## Accepted measurements

Morphology can include body length, mass, scale/tooth counts, trait presence,
ordered wear or developmental scores, and categorical pigmentation. Ecology can
include temperature, precipitation, salinity, habitat categories, or other
explicitly defined measurements. The reader applies each feature's declared type:

| `dtype` | Accepted values and stored type |
| --- | --- |
| `continuous` | Signed decimal/scientific notation, stored as a finite Python float |
| `count` | Nonnegative integer literals containing ASCII digits, stored as Python integers |
| `binary` | One of exactly two declared category labels, stored as a string |
| `ordinal` | A declared label, stored as a string; category order is retained |
| `categorical` | A declared label, stored as a string |

Continuous values use a decimal point and ordinary floating-point precision.
Thousands separators, units embedded in cells, underscores, nonfinite values,
overflow, and nonzero values that underflow to zero are errors. Counts reject
negative values, fractions, scientific notation and signs; leading zeros are
accepted. Counts retain Python integer precision, subject to the interpreter's
integer-parser limit. JSON consumers must preserve large integer precision.
Zero is an observed value. Binary/ordinal labels such as `"0"` and `"1"` remain
strings; the reader does not choose a numeric encoding or assume ordinal spacing.

Features retain their `unit`, declared categories/order, and `role` (`feature`
or `covariate`). At least one analytical feature is required. Units are metadata,
not a physical-plausibility check or automatic conversion. Covariates such as sex,
developmental stage or observer remain explicit; ingestion performs no adjustment.

## Columns and missing values

By default, a table contains an `observation_id` column plus exactly the declared
feature/covariate columns. Headers can appear in any order. Configure nondefault
source roles using the optional modality-level `tabular_options` object:

```json
{
  "observation_id_column": "measurement_id",
  "unit_id_column": "specimen_id",
  "missing_values": ["", "NA"],
  "ignored_columns": ["notes"]
}
```

Defaults are `observation_id_column="observation_id"`, `unit_id_column=null`,
`missing_values=[""]`, and `ignored_columns=[]`. Options apply only to morphology
and ecology; genomic `reader_options` remains a separate SNPio contract.
Omitting `tabular_options` preserves the canonical serialization and hashes of
existing configurations.

Missing tokens are exact and case-sensitive, apply only to feature/covariate
cells, and become `None` in Python / `null` in JSON. `NA`, `NaN`, `?`, and sentinel
numbers are not automatically missing. A category named `NA` is an ordinary state
unless configured otherwise. A token cannot simultaneously be a declared category
and a missing value. An empty missing-token list disables blank-cell missingness.
Custom numeric sentinels require scientific care because the reader trusts the
explicit declaration.

All declared columns, including ignored columns, must be present. Undeclared,
missing, empty, padded or duplicate headers are errors. Column roles cannot
overlap. Ignored values remain in the staged source copy and are excluded from
typed analytical output. Extra taxonomic labels are never automatically selected
as predictors; benchmark truth should remain a separate evaluation input.

Sources must be UTF-8 (an initial BOM is allowed). CSV uses commas, TSV uses tabs;
delimiter inference, comments and locale conversion are absent. Standard quoted
delimiters, doubled quotes and embedded newlines are supported, including CRLF
files. Malformed quotes, invalid UTF-8, wrong-width rows, and blank records fail.
Identifiers and observed values are case-sensitive and are not trimmed. Padded
values and NUL in identifiers/observed values are rejected. Error messages identify
the modality and physical line where a record ends, plus the feature when relevant.

## Biological links and repeated measurements

Source observation IDs must exactly match the selected modality's declarations;
missing, duplicate, undeclared or empty IDs fail. Mapping uses those IDs, never
row position. Source row order is preserved, while typed value columns follow the
declared feature order. A configured `unit_id_column` must agree with the
declaration for every row. When absent, the configuration supplies unit IDs.

Both kinds accept `specimen`, `population`, or `locality` levels. Repeat readings
may share a unit ID but need distinct observation IDs. Partial unit coverage is
preserved and reported. Shared locality/population links do not establish that
different modalities were measured on the same individuals. Locality environment
is retained at locality level rather than copied to every specimen. These data
are not automatically aligned, aggregated, imputed, scaled, dummy-coded, adjusted
for covariates or supplied to the genomic baseline.

All-missing columns and observations with no observed analytical features are
retained and listed in the summary. Successful ingestion establishes parsing and
link consistency, not statistical readiness. Repeated observations are not
independent biological replicates; later models must address dependence,
measurement error, missingness and unpaired data explicitly.

## API and artifacts

```python
from pathlib import Path

from super_deli import load_config
from super_deli.io.tabular import ingest_tabular

source = Path("examples/tabular/dataset.json").absolute()
loaded = ingest_tabular(
    load_config(source),
    base_dir=source.parent,
    output_dir="/tmp/new-tabular-api-run",
    modality_names=["external_traits", "site_environment"],
)
traits = loaded.tables[0]
print(traits.observation_ids, traits.unit_ids)
print(traits.features, traits.values, traits.missing_mask)
```

`LoadedTabularData` contains a tuple of `core.tabular.TabularTable` records,
the output directory, and the run summary. Tables contain immutable tuples of
observations, units, feature declarations, typed rows and a derived Boolean
missing mask. No NumPy, pandas, SNPio or scikit-learn imports are required.

Each new run directory contains:

- `modality_000/input/<source filename>`: an isolated, byte-verified source copy.
- `modality_000/table.json`: version-one typed data with modality kind/level,
  feature metadata, observation/unit IDs, values and missing mask. Subsequent
  tables use consecutive numbered directories; modality names are never paths.
- `summary.json`: completion marker, selected tables/artifact references,
  coverage/repeated counts, missing counts, all-missing columns/analytical rows,
  effective options, software versions, full configuration snapshot/hash and
  selected-source SHA-256 fingerprints.

Existing outputs are refused. Source files are read without modification.
Failures after creation leave `failure.json`, staged diagnostics and any completed
table artifacts; absence of `summary.json` means the whole import did not
complete. Partial artifacts do not constitute a successful multi-table import.
Source rows are parsed sequentially, but loaded tables and JSON serialization
are held in memory; out-of-core ingestion is not implemented. CLI success returns
0 and errors return 2, with JSON on stdout and error messages on stderr.

The stored `examples/tabular/expected.json` is hand-authored truth for values,
ordering, types, missingness and links. These fixtures are illustrative ingestion
checks. They contain no empirical taxonomic ground truth or simulated species
model, and ingestion produces no delimitation/support/significance result.
