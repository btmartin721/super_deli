# Empirical curation results

One matched morphology–ecology development cohort was curated and independently audited. All 1,980 selected numeric values retain their original source tokens. No statistical models were fitted. The data pass ingestion validation; biological sampling independence and independent taxonomic reference validation remain unresolved.

## Cohort and missingness

| Quantity | Count |
|---|---:|
| Source capture records | 344 |
| Unique individual codes alone | 190 |
| Unique study + individual keys | 344 |
| Complete matched capture records | 330 |
| Excluded for missing declared features | 14 |
| Matched continuous views | 2 |
| Exact numeric tokens audited | 1,980 |
| PCA / GMM / bootstrap / significance fits | 0 |

| Feature | Kind | Missing source values |
|---|---|---:|
| bill_length_mm | morphological | 2 |
| bill_depth_mm | morphological | 2 |
| flipper_length_mm | morphological | 2 |
| body_mass_g | morphological | 2 |
| delta_15n_permil | ecological | 14 |
| delta_13c_permil | ecological | 13 |

Missingness counts overlap; they are not additional excluded specimens. Complete-case selection may reflect the source missingness process; no missing-at-random assumption or correction is inferred. Eligibility uses only the six declared features. Source reference assignments and all covariates remain outside feature tables and do not determine inclusion.

| Published reference label | Source rows | Included rows |
|---|---:|---:|
| Adelie Penguin (Pygoscelis adeliae) | 152 | 141 |
| Chinstrap penguin (Pygoscelis antarctica) | 68 | 67 |
| Gentoo penguin (Pygoscelis papua) | 124 | 122 |

These exact source labels are retained as reference hypotheses; nomenclature is not silently normalized. They do not certify species boundaries or provide an independently adjudicated gold standard.

## Audit and applicability

The independent standard-library audit reconstructed the complete cohort directly from immutable source bytes, its natural keys and declared roles. It checked every numeric token, anonymous identity, reference/metadata join, missingness decision, output checksum and protocol/profile binding. The download recipe independently reproduced the same mirror bytes. Source and earlier checkouts/artifacts remain unchanged.

Paired nesting adults and unresolved cross-season identity prevent a certified independent-individual interpretation. Multiple reference taxa share islands, precluding island aggregation for delimitation. No persistent-individual or nesting-unit correspondence is invented. Grouped predictive screening and scaled-fold bootstrap are inapplicable to this capture-level design. Existing model APIs still require their original explicit independence assertion; this curation command does not make it.

Field labels are present for 330 included captures. Their agreement with future clusters would be descriptive and potentially affected by morphology, sex, year, island and sampling design. The protocol designates this familiar source as a development case. A later empirical species-validation study needs independently supported reference evidence and an appropriate sampling/resampling design. No empirical ARI, selected K, stability, p-value, power or species-probability result is reported here.

## Source provenance and deferred candidates

The exact author-mirror commit, URL, byte length, SHA-256 and Git blob identity are locked in the [profile](../examples/empirical-foundation/profile.json). The authors provide a CC0 license. Direct EDI endpoints returned HTTP 403; older recipe DOIs and newer README DOIs are retained separately because upstream-version equivalence is unverified. The original publisher/source citations and license evidence are retained locally.

The [catalog](../examples/empirical-foundation/catalog.json) also records two metadata-only candidates. The [monitor-lizard study](https://doi.org/10.5061/dryad.tqjq2bwb0) requires audited DArT conversion, landmark/sliding decisions and reference-incorporation assessment. The [songbird study](https://doi.org/10.5061/dryad.0sq68) requires workbook and voucher-join auditing. Neither is represented as downloaded, matched or inference-ready.

The [method and commands](empirical-benchmark-foundation.md), [frozen protocol](../examples/empirical-foundation/protocol.json) and [compact cohort audit](../examples/empirical-foundation/results/cohort-summary.json) describe the deliverables. Full data and derived reference/identity mappings remain local; repository fixtures are entirely synthetic. No clustering defaults, numerical code or historical p-values were changed.
