# Juvenile-skate replacement-source findings

The source audit is complete. The data are compatible with audit-only typed
morphology and fixed-landmark representations, but are **not yet an independently
validated replacement benchmark**. The specimen/voucher crosswalk and
independence-dependent sampling design remain conditional or blocked.

The source is [Loca et al. (2026)](https://doi.org/10.1002/ece3.74305),
with the [CC0 archive](https://doi.org/10.5061/dryad.z08kprrtz). All seven archive
files were obtained through normal public browser downloads and matched Dryad's
sizes and SHA-256 digests: 73,721 bytes. API downloads returned 401 and direct
automated file links returned 403; those failed routes are retained as access
limitations. The licensed article and supplement were retrieved through Europe
PMC's public API. No security restriction was bypassed.

## Computed source accounting

| Quantity | Audited count |
| --- | ---: |
| Metadata specimens | 84 |
| Source labels: common blue / flapper skate | 45 / 39 |
| GenBank molecular records | 85 |
| Candidate voucher links: full disc / colour | 23 of 37 / 20 of 29 |
| Full-disc records with nine fixed landmarks | 37 |
| Colour records / complete six-feature colour records | 29 / 23 |
| Shape–colour overlap / overlap with complete colour | 18 / 15 |
| Anterior-margin / rostrum records | 27 / 25 |
| Original coordinate tokens, including curves | 3,814 |
| Original TPS scale factors | 89 |
| Retained analytical coordinate / nonmissing colour tokens | 666 / 162 |

All 84 source specimens remain declared in the audit manifest. Thirty-six have
neither a full-disc nor a colour observation. There is no automatic deletion of
those specimens or imputation of missing views. All raw geometry and source
membership remain in the local evidence.

Six zero-area bands have missing band greyscale and derived greyscale-percentage
values. Those zeroes and twelve `-` cells are preserved. The two supplied colour
percentages are deterministic functions of other columns, within the declared
numerical tolerance; they are not independent evidence dimensions. No source
imputation, zero replacement or feature-selection rule was adopted.

The supplement lists 53 measurement specimens. Its full-disc and colour flags
match the corresponding files. One anterior-margin record is additional to its
stated flags; the candidate image-stem union across all four files has 54 keys.
The discrepancy and source IDs are retained locally without correction. Curve
image stems are diagnostic candidate links, not certified specimen identities.

## Molecular-reference audit

The stated accession range PZ664735–PZ664819 contains 85 distinct versioned
records. Independent parsing of XML and separately retrieved GenBank text
agreed on accession versions, full specimen vouchers and organism names.

No archive short ID literally equals a complete GenBank voucher. Two observed
voucher-prefix conventions yield 53 unambiguous candidate namespace matches,
whose organism names agree with the corresponding source labels. The remaining
31 metadata IDs are unmatched under that prespecified rule. Vouchers include
zero-padded and alternate identifier forms; no zero stripping or additional
normalization is inferred. These are linkage
checks, not classification accuracy or independent biological validation. The
full ledger retains all 85 molecular records and does not discard the records
outside the archive's metadata-ID set.

The paper reports deceased sampling, COI assignments using diagnostic nucleotides,
and species-separated hauls. The archive supplies no authoritative specimen/haul
or full-voucher crosswalk. Single-locus mitochondrial reference hypotheses also
do not establish infallible species-boundary truth.

## Readiness decisions

- Typed audit ingestion and descriptive source-label reporting: available.
- Fixed full-disc geometry: available as an unchanged representation, without
  alignment/PCA or certified physical scale units.
- Independent reference concordance: conditional on an authoritative voucher
  crosswalk, with unresolved and ambiguous records explicitly accounted for.
- Independent-unit fitting, grouped resampling and independent holdout:
  blocked pending specimen-to-haul links and a justified analysis design.
- Curve/semilandmark use: inapplicable to the current fixed-landmark pipeline.
- New significance, stability, performance or species-probability claims:
  not authorized and not generated.

The archive API reports `visibility=public`, `curationStatus=Published` and
`versionStatus=submitted` for version 8 / resource 460651. All three fields are
retained. The locks identify the actual publicly retrieved files; no equivalence
between those distinct status fields is asserted.

## Verification and next boundary

A separate standard-library auditor reconstructed every source coordinate,
scale factor, reference/metadata/feature join, source length, exclusion and
membership comparison. All prior estimator/default and empirical-schema source
files remain byte-identical; only the version and thin CLI adapter changed
among old runtime modules. Local verification passed 1,656 tests with scientific
extras and 1,108 foundation tests (103 optional-dependency skips), both installed
offline smokes, package checks and byte-identical installed reproduction. All
211 source modules were checked in the wheel/sdist and against Python 3.11
grammar. Final Python 3.11/3.12 CI is verified separately for the published SHA
before recording checkpoint completion.

The next scientific prerequisite is authoritative source/voucher and
specimen/haul correspondence, or another fully documented source if those
records cannot be obtained. Then freeze a dependence-aware empirical point
protocol. This audit does not authorize contacting authors or fitting models.
Full evidence remains local under
`~/.codex/outputs/super-deli/replacement-empirical-20261025/`.
