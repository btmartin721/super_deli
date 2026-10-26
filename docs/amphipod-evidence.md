# Amphipod sampling and reference evidence

The separately versioned `TraitEvidencePlan` assesses the completed
[trait-source audit](amphipod-audit.md) without changing that audit, previous
schemas, models, defaults or historical p-values. It is an offline evidence
assessment and grants no fitting permission.

```bash
super-deli assess-trait-evidence examples/amphipod-gates/plan.json \
  --prior-plan examples/amphipod-audit/plan.json \
  --source-dir /absolute/path/to/prior-sources \
  --prior-audit-dir /absolute/path/to/completed-trait-audit \
  --evidence-dir /absolute/path/to/reviewed-evidence \
  --output-dir /absolute/path/to/new-assessment
```

The declaration binds the exact prior specification/checksums and immutable new
source snapshots. Every prior artifact is checked against a complete replay of
the unchanged source-audit workflow; editing values and recomputing checksums
cannot conceal a semantic mismatch. Inputs are rechecked before completion.
Failed assessments retain a failure record and no successful top-level summary.

The source set contains the [2024 trait paper](https://doi.org/10.1038/s41597-024-03020-w),
its four linked Figshare records, the final [2025 molecular paper](https://doi.org/10.1111/fwb.70000),
all eight [public supplements](https://doi.org/10.5281/zenodo.14441819), and the
[public catalogue access statement](https://db.subbio.net/). The final molecular
paper remains distinct from the preprint cited in the earlier trait publication.
Published inventory sizes/MD5s and source SHA256s are checked independently.
PDF/DOCX review text has separately locked derivatives; the independent audit
reproduces those extractions. The rendered catalogue access statement is a
curator-observed public snapshot. No private catalogue record is accessed.

`io.evidence_workbook` streams the unusually formatted reference worksheet while
sharing the original strict cell decoder. It bounds expanded ZIP content to
128 MiB, streamed XML to 96 MiB, physical row coordinates to Excel's 1,048,576,
and populated rows to the first 10,000 worksheet positions. At most eight sheets
are accepted. Empty formatting rows are counted separately and never become
specimens. Stored values/types, formula caches, row attributes, date systems,
styles and merged ranges are retained. Merged cells are not filled down. Formulas,
links, macros and supplied R code are never executed. The original trait reader
retains its existing limits and behavior.

Reference tables are reconstructed by explicit sheet, header and field roles.
All populated source cells and every declared table record remain local, including
missing accessions, duplicate identifiers, composite-marker donors and unused
context tables. Original scalar strings and partial dates remain unchanged.
Tree tips and regional MOTU lists provide label context only; no tree is inferred
and a tip match never supplies a specimen crosswalk.

A specimen association requires the existing literal GenBank voucher/COI check or
an unambiguous explicit voucher/COI pair in the published phylogeny table.
Known contradictions take precedence over an attractive matching claim. Source
cautions and repeated accession rows withhold use. A sequence-name token or a
voucher-only candidate cannot bridge a missing identity/accession mapping.
Composite phylogeny rows cannot bridge an absent GenBank voucher. Even for an
already corroborated COI link, they do not establish that other markers came
from the same specimen. GenBank qualifiers and prior link statuses are preserved.

Historical workbook MOTUs, accepted study labels and raw PTP/ASAP partitions are
separate evidence. No partition is chosen or used to correct another label.
PTP and ASAP strings inhabit different namespaces; their literal inequality is
not disagreement. The report retains their label-invariant one-to-one or
split/merge context across published records, without computing Super Deli
accuracy, new delimitations or species significance.

The source review documents targeted specimen selection, mixed MOTU-assignment
basis, sequence-based raw delimitations, subsequent integrative review and
reference-informed source imputation. Reviewed claims retain exact source
locators, interpretations, limitations and required evidence. Snapshot integrity
cannot prove a paper's correctness or a reviewer's interpretation. Computed
facts and reviewed scientific claims remain distinct.

Purpose-specific readiness separates ingestion, descriptive labels and documented
molecular-reference links from independent-unit fitting, grouped screening,
resampling, significance and species-boundary validation. Locality/date token
agreements and source syntopy numbers are context; neither supplies a verified
sampling-event crosswalk or certifies unrelated specimens/independent localities.
No cohort, population map, missingness policy, analysis configuration or new
biological independence assumption is created.

Detailed inputs, reference cells, per-record accounting and full checksums stay
local. Public package assets are an exact allowlist of code, documentation, source
locks, invented fixtures and compact findings. New source downloads are bounded
to 16 MiB, retained artifacts to 64 MiB, with at least 8 GiB free. Python support
remains 3.11/3.12; installed smokes block numerical imports and network connections.
No PCA, alignment, clustering, bootstrap, significance analysis or author/custodian
contact occurs. See the [findings](amphipod-evidence-results.md).
