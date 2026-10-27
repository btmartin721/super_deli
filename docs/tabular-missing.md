# Opt-in missing-data policy comparison

```bash
super-deli analyze-tabular examples/tabular-missing/dataset.json \
  --analysis-config examples/tabular-missing/analysis.json \
  --output-dir /absolute/path/to/new-missing-data-analysis
```

`TabularMissingAnalysisConfig` is a separate schema-2 contract dispatched through
`analyze-tabular` and `run_tabular_analysis`. Existing schema-1 configurations,
standalone mean imputation, numerical defaults and historical p-values remain
unchanged. This opt-in workflow supports one continuous morphology or ecology
table, with explicit selected traits and one declared independent biological unit
per row. It does not verify independence or open empirical readiness gates.

The configuration contains the existing prespecified `arms`, `schema_version: 2`
and a `policies` array containing `complete_case`, `mean`, or both. Arm model
settings retain their legacy shape; GMM's base `missing_policy` remains `error`
in these requested settings. The top-level policy is compiled into the effective
per-arm preprocessing and recorded explicitly in the versioned manifests/native
provenance. A policy cannot be silently inferred from file contents.

Complete cases retain rows observed at every selected trait. Mean treatment
retains rows with any selected measurement and estimates feature means using
only that training fit's observed cells. Fully unobserved rows are excluded with
an explicit reason and receive no cluster assignment; they are not noise.
Source-declared missing tokens remain distinct from malformed or nonfinite
measurements. Metadata/reference labels and unselected columns cannot influence
eligibility, fill values or model selection. Original bytes, typed values,
identities, order and masks are preserved alongside policy-specific raw cohorts.

A selected feature with no observed values fails the new fit. Existing logged
zero-variance exclusions and fixed-rank checks remain in effect; a projected unit
with no observed retained features can also fail a bootstrap draw. These stricter
opt-in guards do not change legacy standalone behavior. No missing column is
filled with zero or removed opportunistically to make the new policy succeed.

Both families reuse the existing standard-scaling/full-SVD PCA implementation.
Mean-filled values have zero centered residual and remain in the standardization
denominator. Scales are not estimated by pretending imputed cells were additional
observations of their original unknown values. Imputation can change variances,
correlations, distances and inferred group structure; even random missingness
requires empirical sensitivity checks for these clustering heuristics.

Every GMM bootstrap refits observed means, scales, feature checks, PCA and its
complete K search on sampled whole rows. Original eligible units are projected
using that draw's training parameters. Native in-bag/OOB denominators and failed-
outcome bounds remain separate. HDBSCAN uses the existing 80% subsampling without
replacement, refits all preprocessing, and assigns only included units. It does
not predict labels for absent units. Density noise is never a shared cluster.

Single imputation supplies point estimates. Conditional stability is not
multiple-imputation uncertainty, species probability or a p-value. GMM conditional
membership and HDBSCAN density strength remain different evidence types. Failed
draws are retained without replacement and never become K=1 or noise. The workflow
creates no significance results and imports no historical ones. Grouped,
integrative, count/categorical, allometric and reference-informed imputation are
outside this interface.

Each policy's two arms share eligible units and preprocessing. Cross-policy
agreement uses explicitly aligned common resolved units and reports original-
source coverage, exclusions and noise. Absolute BIC is not compared across
different cohorts or preprocessing. The report declares no winning model/policy
or species partition. Partial stability retains valid point evidence; failed or
inconsistent point fits are unavailable.

The local root contains the immutable manifest/source snapshot, typed ingestion,
policy eligibility/missingness ledgers, native results and a report. Versioned
`training-context.json` and compressed arrays retain every training matrix/mask,
means, scales, PCA and successful selected-model parameters, including per-draw
failures. Evidence retention is bounded before fitting with the model's workspace
budget. Full raw/synthetic inputs, masks and fitted arrays stay local.

The frozen validation uses 60 independent complete datasets across Gaussian,
four-group and curved two-group laws at N=64/256, with five prespecified masks
and unchanged model settings. It compares complete cases and observed-mean
imputation with complete-data simulation references. Ten datasets per cell give
bounded sensitivity evidence rather than a broad performance guarantee. MAR/MNAR
stress definitions are simulator assumptions, not mechanisms inferred from
amphipod data. Generating labels enter post-fit evaluation only.

The reproducible phases are `examples/tabular-missing/run.py preflight`, `audit`,
`gate` and `main`. Main requires an independently audited source-bound preflight
and measured storage gate. Planned budgets are 408 preflight and 1,836 main
pipeline slots, including native resampling/replays; prerequisite-blocked and
failed slots remain explicit. The independent audit checks raw generation/masks,
training statistics, mixture densities/BIC, HDBSCAN on audited training scores,
projection, seeds, support denominators and exact replay arrays. It does not
import the missing-data selection or orchestration implementation. HDBSCAN is
checked with its declared library estimator, not independently reimplemented.

Keep detailed evidence within the 4 GiB local cap, at least 8 GiB free and at most
four single-thread study workers. Preserve defaults and prior evidence; public
assets contain code/docs, invented small fixtures, figures and aggregate findings.
Python support remains 3.11/3.12. No empirical source analysis or author contact
is authorized by this checkpoint. See [the validation findings](tabular-missing-results.md).
