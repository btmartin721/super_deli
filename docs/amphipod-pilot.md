# Fit-free amphipod pilot protocol

```bash
super-deli assess-trait-pilot examples/amphipod-pilot/plan.json \
  --prior-plan /absolute/path/to/trait-audit-plan.json \
  --evidence-plan /absolute/path/to/trait-evidence-plan.json \
  --source-dir /absolute/path/to/locked-trait-sources \
  --prior-audit-dir /absolute/path/to/completed-trait-audit \
  --evidence-dir /absolute/path/to/locked-reference-sources \
  --prior-assessment-dir /absolute/path/to/completed-evidence-assessment \
  --output-dir /absolute/path/to/new-pilot-feasibility
```

The separately versioned `TraitPilotProtocol` and
`super_deli.workflows.trait_pilot.assess_trait_pilot` freeze a future descriptive
pilot design and assess raw morphology rosters without fitting. Existing
source audit/evidence schemas, model configurations and p-values are unchanged.
This workflow requires no numerical libraries or network access. Every prior
artifact is checksum checked and the entire evidence assessment is reconstructed
from locked sources; a rewritten checksum cannot legitimize altered semantics.
A failed run retains `failure.json` and withholds the completed summary.

The [trait workbook](https://doi.org/10.6084/m9.figshare.24160872.v1) contains raw
measurements. The [trait paper](https://doi.org/10.1038/s41597-024-03020-w) also
provides a separate phylogeny-informed imputation workflow; its outputs are
excluded here. The [molecular study](https://doi.org/10.1111/fwb.70000) supplies
reference hypotheses through the previously audited public records.

Four rosters are fixed before counting their membership or reference coverage:
19 core traits versus 24 somatic traits, separately for source-labelled male and
female Niphargidae. The core panel contains body length, two antenna lengths,
eight gnathopod measures, two coxa depths and six pereopod measures. The secondary
panel adds five uropod measures. Egg diameter is reproductive-state-specific and
is excluded; egg counts and spine categories are not continuous panel features.
Uropod measurements have source-documented sparse coverage and sexual dimorphism.
The complete ordered trait lists are retained in `frozen-design.json`.

Unique audited vouchers and complete, finite, strictly positive selected raw
measurements are required. Selected formula cells, even with cached values,
and source numeric issues are excluded. Malformed/uncertain tokens, missingness,
nonpositive values and formula provenance remain unchanged in local ledgers.
No decimal repair, imputation, recalculation, scaling, ratios, allometric
correction or age inference occurs. Source male/female labels do not independently
verify adulthood; subadult, juvenile and unknown labels are excluded. Body length
remains a feature, so any future structure could reflect size rather than species.

Selection reads no MOTU, species, accession, reference partition or locality/date
value. All four membership decisions finish before reference attachment.
Reference conflicts and missing links limit evaluation without changing the raw
morphology cohort. PTP, ASAP, accepted study labels and historical MOTUs remain
separate; no method or specimen subset is chosen for attractive concordance.
The future non-executable design records the existing two-PC/full-BIC GMM and
HDBSCAN point settings, with zero resampling/significance draws. It does not
instantiate a model configuration or assert independent units.

Local outputs include all source-row decisions for each roster, overlapping
reason flags, a fixed-order exclusion waterfall, raw selected tokens, full
reference evidence, formula provenance, descriptive trait/context/reference
accounting, elementary size bounds, checksums and a readiness report. Per-trait
exclusion counts cover all source rows, including rows excluded for identity,
family or sex. First-exclusion counts are disjoint; overlapping counts are not.
The 19- and 24-trait rosters overlap and cannot be added as independent samples.
Reference label frequencies and locality/date tokens are descriptive context,
not prediction accuracy, sampling events, independent localities or effective N.
No real specimen rows or executable clustering configuration are exported publicly.

Independent audit is available as
`super_deli.benchmarking.trait_pilot_audit.audit_trait_pilot(output_dir)`.
It reconstructs all membership flags from raw cell tokens, verifies complete
accounting, recomputes every roster summary field and checks the closed gates
without importing the new protocol, selection or reporting implementation.
It reuses the unchanged source audit's identifier-syntax decision; independently
verified voucher counts and hashes do not establish biological independence.

Ingestion and descriptive rosters can be available while fitting, grouped
screening, resampling, significance and species-boundary validation remain blocked.
This pilot has no measured continuous ecology. No ecology, sampling unit, popmap
or private catalogue crosswalk is manufactured. The checkpoint stops with its
explicit no-go outcome for inferential analysis; it does not contact authors or
start another source search. Detailed evidence stays local within a 128 MiB
retained-artifact cap, with at least 8 GiB free; public assets contain only code,
source locks, invented fixtures and small aggregates. See the
[findings](amphipod-pilot-results.md).
