# Replacement empirical source audit

This fit-free checkpoint audits the juvenile-skate study, its CC0 archive,
molecular-reference records and sampling evidence. It preserves all existing
estimators, defaults, p-values and empirical schemas. No alignment, PCA,
clustering, imputation, sliding, bootstrap, significance calculation or execution
of the source R script occurs. The source's classifiers and published accuracy
are not Super Deli validation results.

## Narrow source contract

`core.replacement_audit` defines an immutable, separately versioned source-audit
plan: exact role/filename/URL/size/SHA-256 locks for twelve source artifacts,
explicit source-to-GenBank taxon names and candidate voucher namespaces. It
accepts no fitting permission or independence assertion. The dataset license
field is `archive_license_id`; it does not relabel the article's CC BY license.

`io.specimen_audit` inspects the source's exact CSV roles, a bounded TPS subset,
NCBI GBSet XML and the supplement's specimen-membership table. TPS fixed points,
curve sections and metadata are distinct. The input adapter supports the
archive's specific eight-point curve sections for auditing only; it does not
extend the production fixed-landmark reader. Case-sensitive `ID` and `Id` headers
are explicitly handled. Unknown syntax, malformed/nonfinite numeric tokens,
duplicate primary identities and conflicting reference accessions fail with
retained diagnostics rather than being repaired silently.

`workflows.replacement_audit` checks archive inventories against every raw file
lock, accounts for all specimens and all four measurement files, and records
source tables, geometry, exclusions/unmatched records, reference links and
coverage. Reference-record duplicates/ambiguities are not collapsed into a
favored label. Original source values, including zeroes and `-` missing tokens,
remain unchanged. Derived colour ratios are audited as redundant information,
not assumed to be independent measurements.

Source locks verify which bytes were inspected. Human review of the paper,
reference provenance and sampling limitations remains necessary; a checksum or
an observed naming convention cannot certify biological truth.

## Record linkage is separate from reference method

Source labels describe COI-supported assignments, whereas the analytical views
are morphology/shape. Those sequences remain reference evidence only. They are
not a diploid SNP input, an independent genomic benchmark or a new SNPio reader.
All existing SNPio format/popmap behavior remains unchanged.

The workflow compares literal full-voucher identities first. Separately, it
reports candidate matches after removal of two prespecified observed namespace
prefixes. That candidate operation does not use the metadata species label to
choose a record. It does not certify the namespace mapping. A complete,
authoritative voucher-to-short-ID crosswalk is still required before claiming
independent specimen-level reference concordance. Missing, ambiguous and
conflicting records retain their own denominators and evidence tiers.

The review distinguishes distinct specimen records from independent biological
observations. Record uniqueness cannot establish unrelatedness or sampling
independence. No haul is inferred from sample prefixes, no species label becomes
a population/locality, and no photograph becomes another independent specimen.

## Audit-only derivatives

Where literal source correspondence exists, the workflow writes an existing
`DatasetConfig` with all 84 declared specimens, including specimens without one
or both views. It retains all 37 full-disc observations and all 29 colour rows,
including incomplete colour measurements. It does not create a complete-case
analysis cohort or fill absent views with zeroes.

The main configuration uses one long-form coordinate CSV and one typed colour
CSV. Analytical feature files contain no species labels; source labels and
length/sex covariates are separate. Anonymous specimen keys depend only on the
source identity and audit ID. Each full-disc TPS derivative contains one
specimen, retains exact coordinate tokens, `IMAGE` and `SCALE`, and is checked
by the unchanged strict TPS reader. The source `VARIABLES` payload is preserved
in the geometry audit rather than admitted as an analytical feature.

`analysis/tps_manifest.csv` is an audit index, not a new ingestion format. It
explicitly distinguishes the coordinate-CSV observation ID from the TPS image
observation ID. The main dataset uses the coordinate CSV so that all full-disc
records remain one anatomical view.

Numeric scale factors are retained without multiplying coordinates or asserting
physical units. A ruler's units in a photograph do not alone prove the units of
a saved TPS factor. Physical measurements therefore stay gated. The two curved
views remain archived/audited outside fixed-landmark readiness.

## Offline command

With the preserved local source bundle and this checkout:

```bash
replacement_run="$(mktemp -d "${TMPDIR:-/tmp}/super-deli-replacement.XXXXXX")"
super-deli audit-replacement-empirical examples/replacement-empirical/plan.json \
  --source-dir "$HOME/.codex/outputs/super-deli/replacement-empirical-20261025/locked-sources" \
  --output-dir "$replacement_run/audit"
```

The command never fetches a URL or accepts replacement source bytes. Public URLs
are source locators; the supplement locator's fragment identifies the extracted
DOCX member of the Europe PMC container. Saved public-source snapshots are the
reproducibility record when live metadata change. The 85 versioned accessions are
retained in the local reference ledger, not copied into feature tables.

The output directory must be new and separate from sources. Late failures leave
`failure.json`; completion is renamed into place only after source rechecks and
checksums. Invented fixtures exercise the same CLI offline, with numerical
imports and network connections blocked.

## Publication boundary

GitHub contains explicit source locks, citations, compact aggregates and invented
fixtures. Full raw source files, specimens, sequence records, crosswalks and
review evidence stay local under the approved 64 MiB new-artifact cap, with at
least 8 GiB free. No author was contacted. The package manifest adds exact public
asset paths and retains its existing explicit allowlist.

See the [audit findings](replacement-empirical-audit-results.md). Any future
point evaluation needs a separately frozen design after its relevant sampling
and reference prerequisites are established. This source audit opens no earlier
penguin gate and generates no runnable clustering configuration.
