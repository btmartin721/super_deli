# Specimen, voucher and haul correspondence

The `assess-correspondence` command is a separately versioned, fit-free evidence
assessment. It preserves the completed replacement-source audit and all older
empirical schemas. It does not authorize an empirical analysis or relax a model
assumption. The [skate findings](skate-correspondence-results.md) remain unresolved.

## Command

```bash
super-deli assess-correspondence assessment-plan.json \
  --prior-plan previous-audit/specification.json \
  --source-dir locked-sources \
  --prior-audit-dir previous-audit \
  --evidence-dir evidence \
  --output-dir new-assessment
```

The prior specification fingerprint, complete prior artifact inventory and every
new evidence snapshot are bound by SHA-256. The workflow independently
reconstructs prior reference links from source metadata and GenBank records.
An existing output directory, unsafe overlap, altered input or late evidence
mutation is an error. Late failures retain `failure.json` without a completed
`summary.json`. Inputs and historical outputs are never overwritten.

## Documentary records and source review

`core.correspondence` holds immutable declarations; `io.correspondence` reads
narrow documentary CSV and official DATRAS XML; `workflows.correspondence`
reconciles records and writes evidence artifacts. Existing parsers, prior audit
checks and provenance helpers are reused. No scientific dependency is required.

A reviewer must classify evidence sources and retain issuer/context, URL,
license, provenance group, source locator, exact snapshot and limitations.
Hashing a document verifies its bytes, not its authority or truth. The program
cannot authenticate an issuer or certify a curator's source-role judgment.
Database mirrors are retained with shared provenance rather than counted as
independent biological corroboration. Evidence collection is external to this
offline workflow. No contact or email operation is implemented.

Documentary voucher tables contain `specimen_id,photo_id,voucher,accession_version`.
Documentary haul tables contain `specimen_id,photo_id,fish_id,cruise_id` plus the
complete declared haul key: `Survey,Year,Quarter,Country,Ship,Gear,SweepLngt,StNo,HaulNo`.
Headers may be reordered but must contain exactly the declared columns. Optional
photo, sampling and cruise fields can remain empty; absence is preserved.
A source-provided photo ID is recorded as declared, without claiming independent
image verification. Blank templates contain no evidence. Source declarations
must refer to reviewed documentary records, not an analyst's guessed assignments.

The catalogue adapter is deliberately restricted to IE-IGFS 2021 quarter 4.
Its composite key is conservative within this scope, not a universal assertion
about all ICES identifiers. Original official tokens, including fixed-width
character padding and leading zeroes, are preserved and compared literally.
No stripping or inferred alias normalization is performed. Short archive and
voucher identifiers still require nonempty unpadded strings. `FishID` is scoped
to its full haul key for collision checks; it is never equated to an archive ID.

## Outputs and decisions

All specimens and reference records remain accounted for. Each specimen keeps
literal/source-documented correspondence, candidates, unresolved links,
ambiguities and conflicts separately. Duplicate documentary rows remain
ambiguous. Accession/voucher disagreement, reused references/photos or repeated
sampling numbers across specimens within a haul remain conflicts. Claims outside
the source cohort are retained rather than silently discarded. Haul links must
match an exact catalogue key; presence in the catalogue alone creates no link.

Coverage can be complete, partial or unresolved for each kind of correspondence.
Even complete documentary mapping does not open independent fitting, holdout,
resampling, significance or species-boundary validation: those require a separate
scientific design. Source-label composition is descriptive and limited to
resolved hauls; an unlinked catalogue contributes no linked groups.

The local output includes complete specimen/reference/haul ledgers, original
haul fields, field definitions, search scope, checksums, blank documentary
schemas and a per-specimen records-needed checklist. These are local review
artifacts. The checklist is unsent, has no recipient and is not permission to
contact anyone. This checkpoint contacted nobody. Email requires an explicit
user request, and the study authors are not being involved.

## Verification

Synthetic tests cover complete, partial, absent and conflicting documentary
records, composite haul scope, literal padding, label/row-order independence,
immutable sources, failure retention and legacy compatibility. The installed
smoke uses invented records with numerical imports and socket connections
blocked. Tests and packaging retain Python 3.11/3.12 support. Only explicit public
asset paths are packaged; full real inputs, crosswalks and evidence stay local.
