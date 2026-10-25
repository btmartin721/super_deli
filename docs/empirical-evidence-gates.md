# Empirical sampling and reference-evidence assessment

This checkpoint adds a fit-free review of the locked capture-record development
case. It preserves all estimators, defaults, existing p-values, schema-1 curation
protocols and future point settings. It creates no new analysis cohort and runs
no PCA, clustering, resampling, simulation or significance analysis.

## Evidence and computation have different roles

`core.empirical_evidence` defines a separately versioned assessment bound to the
exact source SHA-256, complete feature/identity profile and unchanged curation
protocol. Each reviewed claim records source references, a locator,
interpretation, limitation and required evidence. Snapshot locks distinguish
usable content from HTTP failures and browser challenges. An issue discussion
can flag a problem but cannot alone document a biological design. A successful
HTTP status does not establish usable evidence.

The assessment schema deliberately supports this bounded development review
only. It accepts no independence assumptions, fitting permission or arbitrary
readiness booleans. Unsupported promotion of a claim to verified independence
is rejected. A different sampling design requires a new reviewed protocol;
changing a status string is insufficient.

`workflows.empirical_assessment` reuses source locking and exact row parsing,
then verifies every curated artifact checksum. It separately reconstructs the
analytical manifest, observation links, feature roles, exact numeric tokens,
reference/metadata joins, full source index and exclusion accounting. Thus
rewriting a checksum manifest cannot conceal altered analytical or reference
content. All 344 rows are accounted for; the complete cohort remains 330 rows.
The workflow checks reviewed snapshot hashes, retains access limitations,
rechecks inputs at completion and writes only to a new output directory.

Hashes prove which bytes and declarations were reviewed. They cannot prove that
a paper is correct or that a reviewer's interpretation establishes biological
independence. Computed source facts and reviewed scientific claims are separate
artifacts, and the report explicitly retains that distinction.

## Purpose-specific decisions

| Use | Decision | Reason |
| --- | --- | --- |
| Typed ingestion | Available | Source/curated correspondence verified |
| Describing published field labels | Available | Development references with explicit limitations |
| Independent-unit point fitting | Blocked | Independent sampling units not established; fit-free scope |
| Grouped predictive screen | Inapplicable | Capture-level design has no verified group correspondence |
| Resampling | Blocked | No authorized independent resampling units |
| Significance | Blocked | No approved empirical null for this sampling design |
| Independent species-boundary validation | Blocked | Neither independent units nor specimen-level reference adjudication established |

Available descriptive use allows reference counts and provenance reporting. It
does not authorize model fitting or imply that agreement statistics have been
computed. Source-code patterns are never parsed into nests. No seasonal, sex,
reference-label or model-outcome filtering is introduced. Islands cannot be
substituted for independent taxonomic groups when reference taxa coexist there.

## Required next evidence

A sampling decision needs an authoritative capture-to-nest mapping, its numbering
scope across species/islands/seasons, and persistent bird/recapture identity, or
a documented independently sampled replacement cohort. With those records,
a later protocol can justify how to handle paired adults and repeated captures.
A proposed single-season cohort still requires its own sampling justification.
No one-adult-per-nest cohort is inferred or created here.

Independent reference validation additionally needs specimen-linked genetic or
voucher adjudication, provenance and ambiguity/missing-label accounting,
independent of the fitted measurements. Published taxonomy and a separate
phylogeny supply taxonomic context rather than a crosswalk for these captures.
The familiar development data are not a blinded or near-boundary validation.

## CLI and local evidence

```bash
super-deli assess-empirical examples/empirical-foundation/profile.json \
  --source /absolute/local/source_001.csv \
  --protocol examples/empirical-foundation/protocol.json \
  --evidence examples/empirical-gates/evidence.json \
  --curated-dir /absolute/local/curated \
  --evidence-dir /absolute/local/reviewed-snapshots \
  --output-dir /absolute/local/new-assessment
```

`--curated-dir` is the unchanged completed `curate-empirical` output.
`--evidence-dir` contains the reviewed `<source_id>.txt` snapshots with the exact
published fingerprints, including a retained challenge body when present.
Unavailable sources with no body require no snapshot. This command is offline;
it does not fetch URLs, accept replacement bytes or execute external code.
Dynamic web pages may no longer reproduce old snapshot hashes; the local
archived snapshots are the reproducibility record. Live checks require a new
review, not silent fingerprint updates.

Outputs are `reviewed-evidence.json`, `input-integrity.json`, `source-facts.json`,
`summary.json` and `checksums.json`. Completion is published only after checksums
are written. A late failure retains `failure.json` without a completed summary.
No runnable clustering configuration is generated.

The independent audit reconstructs identities/counts/joins directly with the
Python standard library and checks source passages, access diagnostics and gate
reasons without calling the assessment's decision function. Synthetic tests and
installed offline smokes block numerical imports and network connections.
Python support remains 3.11/3.12.

Detailed source/evidence snapshots and row-level inputs remain local. GitHub and
source distributions contain only explicit allowlisted metadata, docs, small
aggregates and entirely invented engineering fixtures. The approved new-evidence
cap is 64 MiB, with at least 8 GiB free. No author was contacted.
