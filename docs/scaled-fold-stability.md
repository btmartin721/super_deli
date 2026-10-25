# Optional scaled fold sizes for shared predictive stability

This separately versioned policy changes the minimum independent-locality support
inside the existing five-fold shared-covariance predictive screen. Existing
full-data minima, defaults, schema-1 model/screen/bootstrap configurations and
historical p-values remain unchanged. Users must prespecify this policy; no
automatic comparison or selection between policies is supplied.

## Exact rule and scope

Require complete continuous specimen morphology linked explicitly to localities,
one continuous ecology row per locality, group-mean morphology PCA, shared
(`tied`) covariance, AIC and five original-ID folds. The original model has no
internal bootstrap (`n_bootstrap: 0`); draw settings remain separate. Missing
features/views, other targets/modalities/families and new significance calibration
are outside this policy's scope.

For a point analysis or bootstrap draw, let U be its count of distinct original
localities and T the distinct count in one training fold. If the declared
full-data minimum is m (at least two), apply:

```text
fold minimum = max(2, (m * T + U - 1) // U)
```

The integer expression implements the exact ceiling without floating-point
rounding. Validate nonempty proper training support. Repeated copies add to the
likelihood and preprocessing weights, not independent locality counts. At N=64
with m=5, a 51-locality training fold uses four and a 52-locality fold uses five.
At N=256, 204 uses four and 205 uses five. Actual bootstrap support determines
each draw's thresholds. No generating label, candidate component size or held-out
feature/outcome selects this rule. Full-data candidate eligibility stays m.

Preserve all raw members, specimen counts, draw multiplicities and the original
five-fold partition across resamples. Copies cannot cross folds. Every training
fold refits ingestion, standardization, view PCA, explicit aggregation, fusion,
joint PCA and the full feasible K search. The strictly positive pooled
occurrence-weighted validation gain and lowest-full-data-AIC rule remain intact.
Newly feasible candidates are actually fitted; no saved or invented parameters
are substituted. Optimizer starts, seeds, convergence, tolerance and regularization
remain unchanged. Numerical failures retain their reasons and unavailable outcomes.

## Architecture and artifacts

`models.predictive_fold_size.PredictiveFoldSizePolicy` is a pure configuration/
integer-arithmetic layer. The optional `fold_size_policy` keyword is accepted by
`run_predictive` and `run_predictive_bootstrap`. When absent, the legacy paths and
serialized outputs retain their original contracts.

Legacy `IntegrativeConfig` keeps its original minimum-size validation, including
the guard above joint PC count. The opt-in path uses a separate
`PredictiveFoldMixture` settings object for the shared-mixture stage. Preprocessing
retains all global dimension/rank checks; the scaled support threshold controls
training feasibility and component eligibility. No general model validation or
full-covariance constraint is relaxed.

Each fold's `model.json` retains its original full-model declaration and optimizer
seed. Before preprocessing/fitting, `fold_support.json` records U, T, m and the
effective fold minimum; `run/mixture_settings.json` records the effective mixture
settings. These sidecars distinguish the declared full-data constraint from the
applied training constraint. They are not replacement integrative-model JSON.
Empty sampled folds retain unavailable thresholds and failure records.

Point/full/fold candidates, held-out densities, selected models and all-original
projections remain auditable. Existing own-point stability outputs preserve
K counts, coassignment, cluster recovery/merging, conditional Monte Carlo
precision, coverage and failure bounds. OOB pair frequencies require at least
ten successful both-OOB draws. The paired benchmark also evaluates the new
procedure against the unchanged legacy point partition, separately from its
own point. Stability of one component is not recovery of separate groups.

## API and CLI

```bash
super-deli cluster-integrative-predictive dataset.json \
  --model-config examples/predictive-stability/model.json \
  --screen-config examples/predictive-stability/screen.json \
  --fold-size-policy examples/scaled-fold-stability/policy.json \
  --output-dir scaled-point
super-deli cluster-integrative-predictive-stability dataset.json \
  --model-config examples/predictive-stability/model.json \
  --screen-config examples/predictive-stability/screen.json \
  --bootstrap-config examples/predictive-stability/bootstrap.json \
  --fold-size-policy examples/scaled-fold-stability/policy.json \
  --output-dir scaled-stability
```

The policy file requires exactly `schema_version: 1` and
`policy: "proportional_distinct"`. Omitting the flag preserves the legacy rule.
All output directories must be new. The policy has no significance-calibration
interface; existing tests do not calibrate this predictive procedure.

## Bounded validation and resources

The approved study fixes five fresh independent datasets in each of eight
alternative/control/stress cells, paired legacy/scaled analyses, and 99 whole-
locality draws per policy: 48,000 candidate searches. Ten archived legacy
point/first-two-draw replays add 180 exact searches. A separately seeded full-
settings one-per-cell probe adds 9,600 searches before the measured resource gate.
There are no truth-initialized fits, extra optimizer arms or significance draws.

Use `benchmark-scaled-fold-stability`, `analyze-scaled-fold-stability`,
`preflight-scaled-fold-stability` and `replay-scaled-fold-stability` for the
explicit study commands. `examples/scaled-fold-stability/check_study.py` verifies
all archive/member bytes, IDs, thresholds, component sizes and gates with the
standard library. The numerical auditor independently reconstructs raw laws,
PCA/projections, shared densities, criteria, selection and stability. It also
compares paired scores and common-candidate parameters exactly; only support
eligibility/feasibility changes. The allowed candidate set cannot shrink when
both workflows are available, but selected K and biological recovery need not
increase.

The measured gate includes archives and manifests, replay allowance, 25% growth
and 128 MiB reserve. Require a 6 GiB new-evidence cap, at least 8 GiB free and
at most four single-thread numerical workers. Preserve failures without replacement.
Detailed raw data, fitted arrays and private indexes remain local; publish only
code, fixed designs, documentation, figures and compact aggregates.

Original datasets are the independent validation units. Five per cell supports
descriptive paired effects/ranges, not general performance or false-positive
claims. Draws/folds repeat measurements. Exact binomial intervals describe only
unadjusted conditional Monte Carlo precision; they are not species probabilities,
simultaneous population confidence or new p-values. Stress departures are not
species or type-I error evidence. Recovery, control overpartitioning, failures
and OOB coverage must all accompany any apparent stability improvement. See the
[results](scaled-fold-stability-results.md).
