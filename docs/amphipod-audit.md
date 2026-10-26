# Fit-free groundwater-amphipod audit

This separately versioned audit checks the public CC0 workbook accompanying
[the functional-trait data paper](https://doi.org/10.1038/s41597-024-03020-w),
[versioned data](https://doi.org/10.6084/m9.figshare.24160872.v1) and
[source collection](https://doi.org/10.6084/m9.figshare.c.6841665.v1). The screen
selected traceability and compatibility with continuous-trait ingestion; it used
no Super Deli clustering results. Ant, foraminifer and shrew candidates were
screened with their own remaining compatibility or correspondence limitations.
This was a bounded source search, not an exhaustive dataset ranking.

```bash
super-deli audit-trait-source examples/amphipod-audit/plan.json \
  --source-dir /absolute/path/to/locked-snapshots \
  --output-dir /absolute/path/to/new-local-audit
```

The schema-1 `TraitAuditPlan` binds exactly seven local, flat, SHA256-locked
snapshots: workbook, archive inventory, collection inventory, publication,
GenBank XML, GenBank text and reviewed screen. The XML and text may be declared
unavailable without converting missing evidence to an assignment. Snapshot
hashes prove which bytes were reviewed, not that biological assertions are true.
The text mirror receives a separate independent audit in this checkpoint;
the reusable workflow parses the XML and retains the text source lock.

The bounded standard-library OOXML reader preserves explicit worksheet/cell
coordinates, all raw scalar tokens, shared strings, stored types, style indices,
formulas and cached values. It reads only declared worksheets/styles/strings;
embedded ink, drawings and inert printer settings are inventoried without
execution. Macros, unsafe paths and external workbook relationships are rejected.
Only NCBI's literal public GBSet DTD declaration is stripped for safe parsing;
it is neither downloaded nor executed. Source bytes remain unchanged.

All 45 columns have declared dictionary roles. Only the 25 continuous traits in
millimetres enter the ingestion representation. Source MOTUs/taxon names,
accessions, geography, habitat categories, sex, people, egg counts and spine
categories remain metadata. No taxon label is learned, corrected or promoted to
reference truth. Missing, malformed and nonpositive tokens remain explicit;
invalid numeric rows are excluded only from the typed representation. There is
no imputation, recalculation, scaling or statistical cohort selection.

Unique literal vouchers yield stable neutral derived identifiers. Missing or
duplicate vouchers never become specimens through technical row numbers or the
nonunique legacy individual-number field. The private source index retains exact
row correspondence. The raw `analysis/morphology.csv` and existing-format
`dataset.json` demonstrate structural ingestion, including explicit missingness;
that directory name does not authorize a model fit or certify a cohort.

Each valid accession is checked for unique use, version, returned record,
source voucher and COI annotation. All source qualifiers are retained, including
non-voucher identifiers which are never treated as voucher aliases. A literal
COI/voucher link establishes traceability only. Source swap/conflicting-MOTU
cautions remain visible and withhold unflagged use; matching taxon strings are
descriptive, not evidence of independent species truth. COI is reference evidence,
not diploid SNP input to the genomic models.

Exact raw name/latitude/longitude tuples are context descriptors only. Unknown
locations never share a synthetic group. Name/coordinate inconsistencies and
source-MOTU composition are descriptive. Dates preserve original text or Excel
serial/style/date-system information; partial dates are not guessed. Locality
context does not demonstrate independent sampling, permit whole-locality
bootstrap or supply a population map automatically.

Compressed row/cell/reference/context ledgers, dictionaries, formula provenance,
descriptive trait/missingness profiles, ingestion files, readiness and checksums
stay local. Deterministic gzip supports byte-identical installed replay. Failed
runs retain `failure.json` and no successful summary. Outputs cannot overwrite
sources or completed audits; source locks are rechecked after reconstruction.
No numerical package, estimator or network request is required.

This checkpoint changes no production model, default, historical p-value or
previous schema. It performs no PCA, clustering, bootstrap, significance test,
source R execution or author/custodian contact. Empirical fitting requires a
later frozen design addressing identities, sampling, missingness and genuinely
independent reference evidence. See the [results](amphipod-audit-results.md).
