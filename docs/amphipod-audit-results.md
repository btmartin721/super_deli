# Groundwater-amphipod audit findings

The [CC0 version-1 workbook](https://doi.org/10.6084/m9.figshare.24160872.v1)
is better documented for specimen-level continuous-trait ingestion than the
previous sources reviewed. It still does not establish independent species truth
or a statistically independent analysis cohort. This checkpoint is entirely
fit-free; models, defaults and historical p-values remain unchanged.

| Check | Retained finding |
| --- | ---: |
| Workbook records / columns / continuous traits | 1,123 / 45 / 25 |
| Distinct nonmissing vouchers / voucherless rows | 1,037 / 86 |
| Valid distinct COI accessions / literal `check` placeholders | 173 / 2 |
| Retrieved GenBank XML and text records | 173 each |
| Exact voucher-and-COI links / absent record voucher | 162 / 11 |
| Unflagged literal links / linked rows with source MOTU caution | 156 / 6 |
| Rows with malformed trait tokens / nonpositive measurements | 8 / 6 |
| Cached formula cells; formulas evaluated | 1; 0 |
| Raw typed-ingestion rows / missing trait tokens therein | 1,029 / 4,073 |
| Complete raw locality contexts / exact context descriptors | 1,086 / 393 |

The 11 molecular records lacking a voucher qualifier remain unresolved for a
literal voucher check even where they contain other identifiers. The six
conflicting-MOTU cautions remain separate from their factual literal links.
Across the entire workbook, 16 rows carry swap/conflicting-MOTU cautions. The
source's MOTU assignments may also involve morphology/locality-based inference;
DNA availability alone does not make those labels independent validation truth.
No source-defined label was treated as a species boundary.

Eight trait rows contain uncertain decimal tokens or `NANA`. Their exact values
are retained rather than repaired to numeric values or missing markers. Sex-
and family-stratified missingness is recorded without biological imputation.
One stored formula and its cached value are retained without recalculation.
The 1,029-row representation has 21,652 finite and 4,073 missing trait tokens;
it is an ingestion check, not a selected complete-case benchmark cohort.

The [paper](https://doi.org/10.1038/s41597-024-03020-w) reports 314 MOTUs and 389
localities; literal nonmissing workbook fields contain 313 and 390 distinct
strings respectively. These are documented counting disagreements, not silently
corrected taxonomy or sampling units. The 183 distinct nonmissing taxon-field
strings likewise do not establish 183 validated species.

Eight locality names have multiple coordinate-token pairs and three coordinate
pairs have multiple names. Twenty-six exact context descriptors contain more
than one source MOTU label. None is declared an independent sampling group.
Dates include 821 preserved numeric cells, 60 day/month/year strings, eight year
ranges, 60 uninterpreted text values and 174 missing values. Neither dates nor
coordinates were standardized into inferred haul, nest or locality identities.

The independent reconstruction checks every source row/cell, dictionary role,
raw measurement/missing token, formula/cache, identity, accession, XML/text
reference record, literal linkage, locality descriptor and exported ingestion
value. Source and output checksums, deterministic installed replay, offline
smokes and Python 3.11/3.12 tests are part of verification. Full evidence remains
local; this repository contains source locks, invented fixtures and small
aggregate findings only. No author or custodian was contacted.

The next empirical step needs a separately approved, frozen design for a
traceable subset: review the available direct DNA evidence and its actual
independence, declare the sampling unit and trait/missingness policy, and keep
reference evidence out of fitting. This audit authorizes no empirical fitting,
bootstrap, significance test or species-delimitation performance claim.
