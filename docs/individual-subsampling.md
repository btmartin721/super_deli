# Whole-individual subsampling sensitivity

This opt-in CLI/API attaches a separate sensitivity analysis to a successful
native integrative run. It supports complete, matched, one-observation-per-
individual data with full-retention fusion, a prespecified shared-covariance BIC
GMM, ordinary 100 starts and one numerical thread. Genomic data are optional;
selected genomic inputs retain their SNPio-ingested calls and source-locus frame.
Existing ingestion supports VCF, PHYLIP, STRUCTURE and GENEPOP with optional popmaps.
Other existing workflows and legacy configuration bytes are unchanged.

```bash
super-deli subsample-integrative /absolute/path/to/point-run \
  --stability-config examples/individual-subsampling/sampling.json \
  --output-dir /absolute/path/to/new-subsampling-run
```

`point-run` is an existing `cluster-integrative` output or native integrative-arm
subdirectory of an `analyze-matched-multimodal` report. Prespecify `n_init=100` and
`fusion_policy="retain_all_v1"` before fitting that point run. The new command does
not refit, overwrite or change its point partition, conditional memberships,
assignment threshold or existing bootstrap. Choose a new output directory outside
that run. A source snapshot, checksums and numerical draw parameters are retained.

```json
{
  "schema_version": 1,
  "policy": "whole_individual_80_v1",
  "fraction": 0.8,
  "n_subsamples": 25,
  "random_seed": 1601
}
```

All five fields are required in JSON. This version fixes the fraction at 80%; the
count may be 1–1000 and the sampling seed must be an explicit unsigned 32-bit
integer. Each refit samples `floor(0.8*N)` distinct individuals without replacement,
keeping every selected view paired by biological ID. It refits filtering/scaling,
GPA, view PCA, profile normalization, full joint PCA and the complete native GMM
K search. The original distinct-individual minimum is preserved. Sampling uses the
new declared seed; optimizer seeds follow the original point model's native
bootstrap child-seed lineage, allowing paired validation without tuning seeds.

```python
from super_deli.models.individual_subsample_config import IndividualSubsampleConfig
from super_deli.workflows.individual_subsampling import run_individual_subsampling

summary = run_individual_subsampling(
    "/absolute/path/to/point-run",
    stability_config=IndividualSubsampleConfig(n_subsamples=25, random_seed=1601),
    output_dir="/absolute/path/to/new-subsampling-run",
)
```

The offline `report.html` separates candidate K frequencies, actual candidate
eligibility, point-cluster representation and included/omitted pair coverage.
`cluster_support.csv` and JSON record how many distinct source individuals from
each original candidate cluster were sampled, how often that number was below
the existing minimum, included-set Jaccard similarity and exact recovery,
splitting/merging counts. Representation does not determine the refitted groups.

Included-pair coassignment uses successful refits containing both individuals;
attempt-based failed-outcome bounds are separate. Omitted-pair coassignment uses
frozen predictions only when both original individuals were left out. It remains
unavailable if no such comparison exists or a requested omitted evaluation failed;
requested/successful counts and bounds are retained. Original-point agreement on
omitted individuals is agreement with the original partition, not independent
species validation. A separate evaluation cohort is used only in the simulation
benchmark. Candidate partitions include individuals that did not meet the source
run's assignment threshold; source assignment decisions remain unchanged.

These are sensitivity measures for a different perturbation from ordinary
bootstrap. They are not interchangeable confidence intervals, calibrated species
probabilities or significance tests. No p-values are computed. Failures are
retained without replacement or K=1 fallback. Parameter export/source-integrity
failures stop the workflow. Sparse omitted-pair coverage and small original
clusters limit interpretation; stability alone does not establish species validity.

The validation covers complete selected inputs with complete matching across views.
The sampling frame holds existing ingested calls/loci and extracted landmark
coordinates fixed. This workflow does not propagate genotyping, image-marker,
landmark-measurement or missing-value uncertainty. Those require separate models
and validation; no such uncertainty guarantee is supplied here.

Detailed outputs contain private specimen-level evidence and should remain local.
The original source files are checksum-verified before and after computation.
The bounded benchmark fixes 48 independent synthetic parents across N=64/256,
one-population controls, balanced and unequal four-shape-group laws, with 25 paired
bootstrap/subsampling draws and 1,024 independent evaluation individuals/parent.
Main/preflight budgets are 2,552/358 searches, including exact legacy replays.
Counts are fixed by design; generating labels are assessed only after fitting.
No empirical cohort, author contact, model/default promotion or significance change.
