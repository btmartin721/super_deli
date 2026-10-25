# Skate correspondence: bounded public-evidence findings

The mapping checkpoint is complete as an evidence assessment. **The skate
specimen–voucher and specimen–haul crosswalks were not resolved from the checked
public sources.** The candidate remains unsuitable for independent empirical
benchmarking without the missing correspondence and a justified analysis design.
No author, custodian or other person was contacted; no email or form was sent.

## What was checked

The [primary study](https://doi.org/10.1002/ece3.74305), its licensed supplement,
the [CC0 Dryad archive](https://doi.org/10.5061/dryad.z08kprrtz), all 85 versioned
COI records, their ENA mirror, official Marine Institute context, and
[ICES DATRAS services](https://datras.ices.dk/webservices/webservices.aspx) were
reviewed. The search was bounded to these sources and directly relevant public
results; it does not establish that unpublished documentation does not exist.

| Quantity | Finding |
| --- | ---: |
| Archive metadata specimens retained | 84 |
| Versioned COI records checked | 85 |
| Prior candidate voucher correspondences | 53 |
| Prior unmatched metadata IDs | 31 |
| Authoritative specimen–voucher links established | 0 |
| Official IE-IGFS 2021 Q4 haul catalogue records | 157 |
| Authoritative specimen–haul links established | 0 |
| Official field definitions inspected | 174 |
| New reviewed evidence snapshots | 15 |
| Empirical fits / new p-values / bootstrap draws | 0 / 0 / 0 |

The archive remains version 8 / resource 460651; all seven published file sizes
and digests agree with the previous audit. No new crosswalk file is listed.
The 85 refreshed GenBank records are byte-identical to the earlier XML. ENA's
85 EMBL records agree on vouchers, versions, year and broad coastal locality,
with no additional sampling qualifiers. Those mirrors share the same submission
provenance; agreement does not independently identify archive specimens.

The supplement's haul map shows locations and counts without specimen or printed
haul identifiers. No coordinates or bubble counts were used to assign specimens.
The [official field dictionary](https://datras.ices.dk/webservices/DATRASWebService.asmx?op=getDatrasFieldList)
describes `FishID` as a running specimen sampling number; there is no documented
bridge to the archive's short IDs. The public age-based individual queries for
Aphia codes 105869 and 711846 each returned an empty CA array for this survey,
year and quarter. This is a narrow endpoint result, not proof that original
fin-clip, photograph or sampling ledgers are absent. An earlier request used the
wrong species-lookup parameters and returned HTTP 500; it is retained as a request
error, not a negative biological result.

The official catalogue preserves 157 complete composite haul keys and all source
fields, including fixed-width padding. Its 156 distinct station tokens are not
substituted for haul identity. None of those catalogue entries has a verified
archive-specimen link. Species–haul composition therefore cannot be reconstructed
for these specimens. The paper's report of species-separated hauls remains a
source claim, not a newly calculated specimen-level result.

## Remaining evidence and scope

A per-specimen local checklist specifies the missing short-ID/photo-to-complete-
voucher/accession and short-ID-to-original-sampling-number/haul records, together
with provenance and corrections. Header-only schemas are clearly unpopulated.
They have no recipient and will not be sent. No zero-padding removal, prefix
assumption, row-order matching, species matching or guessed haul was promoted.

Correspondence gates stay unresolved; independence, holdout, resampling and
significance are not authorized. No new cohort, PCA, alignment, clustering,
sliding or statistical test was run. Existing defaults, p-values, source readers
and empirical schemas are preserved. Full measurements, sequence records,
identifiers and evidence remain local; public assets contain only code,
source-lock metadata, invented fixtures and small aggregates.

The independent audit reconstructs every real specimen/reference/candidate/haul
ledger and checks all retrieved record scopes and failures. Python 3.11/3.12
verification, package checks and installed offline reproduction are recorded
for the published commit before checkpoint completion.

The next data-dependent step requires authoritative correspondence already
available without contacting the authors, or a better documented empirical
source. A separate checkpoint would then freeze an appropriate analysis design.
