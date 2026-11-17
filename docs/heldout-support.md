# Held-out predictive support and rare-group feasibility

This separately versioned opt-in workflow attaches a predictive diagnostic to an
existing native 80% whole-individual subsampling run. It retains that workflow's
complete matched-specimen, full-retention, shared-covariance BIC and ordinary
100-start boundary. Selected genomic, morphology, individual ecology and/or fixed
landmark views remain paired. Existing partitions, assignment thresholds,
bootstrap/subsampling, legacy schemas, numerical defaults and p-values are preserved.

```bash
super-deli assess-integrative-support /absolute/path/to/subsampling-run \
  --support-config examples/heldout-support/support.json \
  --output-dir /absolute/path/to/new-predictive-report
```

```json
{
  "schema_version": 1,
  "policy": "native_k1_omitted_v1",
  "minimum_coverage": 0.95
}
```

```python
from super_deli.models.heldout_support_config import HeldoutSupportConfig
from super_deli.workflows.heldout_support import run_heldout_support

summary = run_heldout_support(
    "/absolute/path/to/subsampling-run",
    support_config=HeldoutSupportConfig(),
    output_dir="/absolute/path/to/new-predictive-report",
)
```

All three JSON fields are explicit. This policy fixes the reporting coverage at
95%; it is not a significance threshold or confidence level. Choose a new output
directory outside the source. Consumed source records, numerical arrays, identities
and settings are checksum-bound, copied to a private reference snapshot and
reverified after scoring. Existing point and subsampling files are never updated.

For each successful draw the command replays only the native K=1 score head,
using the saved training scores and original settings/seed. It verifies the
recorded K=1 BIC. It scores the training-selected model and K=1 on original
individuals omitted from that training draw. The source draw already refit
filtering/scaling, GPA, view PCA, profile normalization, fusion, joint PCA and
complete component search using training individuals. Those transformations are
frozen for omitted projections; no new component selection or preprocessing is
performed by this command. Verified identical K=1 comparisons have exactly zero
gain; raw floating-point differences are retained separately.

Gain is selected minus K=1 log predictive density in the same frozen projected
score space. It is not raw genomic/phenotypic likelihood. Ingested calls/loci,
selected feature lists and extracted landmarks are a fixed declared frame;
genotyping, landmark-measurement and missing-value uncertainty are not propagated.
SNPio input-format and optional-popmap support remains unchanged.

Average each individual's available omitted gains first, then give covered
individuals equal weight. Repeated omissions are not independent observations.
`individual_scores.csv`/NPZ retain requested and successful exposure counts,
means and unavailable individuals. Never-omitted individuals remain unavailable.
A full-cohort complete-comparison mean is withheld after missing comparisons or
uncovered individuals. Available-case means remain explicitly partial. Missing
log-density gains are unbounded: there is no finite failed-outcome range to fill
in. No failed draw is replaced or converted to K=1. Export/source-integrity errors
stop the workflow; numerical comparisons that fail retain an unavailable record.

The offline `report.html` shows coverage, unavailable comparisons, the covered
mean, qualification and original-cluster representation side by side. A qualified
report requires >=95% individual coverage and no failed requested predictions.
Unqualified reports do not provide negative evidence. Cluster-specific summaries
are descriptive annotations using the original same-sample point labels; they do
not independently confirm those groups. Positive global gain can support some
structure without establishing a particular K or every extra cluster. No observed
advantage does not establish one species. Hidden rare groups may be absent from
the original candidate partition and cannot be identified merely by annotating it.

The separately bounded simulation study contrasts total N with smallest-group
abundance, including a ten-member group that remains at ten while total N grows.
Generating labels are evaluation-only; no stratified sampling or truth-informed
training is used. Original group counts, global ARI and minority Jaccard/precision/
recall are reported together. Predictive reference cohorts are independent finite
simulation checks, not species truth or a formal calibration. Frozen descriptive
go/no-go targets determine whether further selection-filter work is warranted;
no automatic filtering, new default, new p-value, empirical fit or author contact.

Detailed inputs, fitted parameters, predictions and individual IDs remain local.
Only code, invented fixtures, documentation and small aggregates are public.
