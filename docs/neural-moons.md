# Curved-moons neural recovery diagnosis

This versioned archival diagnostic traces two simulated curved groups through
eight noisy traits, preprocessing, neural encoding, latent scaling and density
group selection. It changes no production estimator, training, default, old
schema, empirical gate or historical p-value. Its results are retrospective
mechanism evidence, not a fresh performance or species-validation study.

The source is the completed [neural comparison](tabular-neural-study.md), version
0.56.0.dev0. Bind all 80 main observations and 1,210 workflows, plus the separate
120-workflow preflight. The source inventory, frozen source bundle, generating
law, identities, raw/PCA/neural transforms, reconstruction, replays and support
denominators are audited without fitting. Its original exact execution gate is
unchanged. The new diagnostic validates the archived source bundle independently,
checks numerical dependency/Python/platform versions, and records its own package
version, source and configuration fingerprints. It never rewrites old evidence.

## Fixed comparison

Use the 20 archived moons observations and 40 Gaussian/four-group controls, with
ten independent datasets per law at N=64/256. All three neural seeds (0, 101, 202)
are retained, not selected. The cline remains in the archival audit only.

Nine coordinate representations are prespecified:

1. The two generating coordinates with population-SD standardization.
2. The eight observed traits with population-SD standardization.
3. Exact saved native two-PC PCA/HDBSCAN scores.
4. Encoder coordinates before latent standardization, for each of three seeds.
5. Exact saved standardized encoder coordinates, for each of three seeds.

Items four and five each contribute three representations. Final numeric weights
are evaluated without training. No fresh scientific datasets or new stability or
significance draws are generated. Generating coordinates/labels are oracle
references unavailable in empirical analysis. Representation, dimension and
scaling changes alter the metric; these comparisons cannot isolate every effect
of feature construction and measurement noise.

Each representation receives one HDBSCAN hierarchy with Euclidean distance,
min_cluster_size=5, self-inclusive min_samples=5, alpha=1, epsilon=0, EOM and
allow_single_cluster=true. On that same hierarchy, replay the unchanged baseline,
EOM excluding the root, and leaf selection. No alternative policy refits the
graph or becomes a production option. See the [HDBSCAN API](https://scikit-learn.org/1.6/modules/generated/sklearn.cluster.HDBSCAN.html)
and the [earlier two-dimensional diagnosis](noisy-moons-results.md).

There are 540 point density fits and 1,620 point selection evaluations. The
baseline on native PCA and standardized neural coordinates provides 240 archived
point replays already included. Eight additional baseline fits replay the first
retained moons subsample at each N for all four density arms. Main totals are
**548 density fits and 1,628 selections**. A separate preflight uses the first
observation in each of six cells (54 fits) and the first saved moons subsample at
each N for neural seed zero (two fits): **56 fits and 164 selections**. These
probes/replays are repeated measurements, not extra independent datasets.

## Evidence and interpretation

Retain source identities, fixed weights/coordinates, neighborhood traces,
reconstruction, linkage/condensed trees, branch lifetimes/mass and policy outputs
locally. Full dense matrices remain reconstructible from retained coordinates.
Compare four-nearest-other-unit overlap and distance ranks against source and
eight-trait spaces. Use stable row-ID order for diagnostic neighbor ties and
report cutoff/distance ties. The reused legacy graph summary has its original
sorting convention; its neighbor fraction is distinct in a tied case.

Audit dense core/mutual-reachability graphs, independent MST weights/linkage
cuts, leaf-set condensation, branch integrals, frontiers, labels/strengths,
source replays, normalization, independent NumPy forwards, reconstruction and
output accounting. Equivalent tied-edge trees need not use identical edges.
Library tree-to-label replay complements the independent graph/tree checks;
neither is advertised as a second complete HDBSCAN implementation.

Report K=0/1/2/>2, merging/splitting, coverage/noise, full-unit and assigned-only
ARI separately. Each noise unit is an unresolved singleton for full-unit ARI.
Pointwise exact intervals and dataset SDs are descriptive for this retrospective
archive, with requested/successful/failed counts and unresolved-outcome bounds.
They carry no simultaneous, post-selection, species or general type-I-error
guarantee. Seeds, branches and old subsamples are not independent datasets.
Training reconstruction is not external prediction. Root exclusion can create
spurious partitions in one-group controls; correct K alone is insufficient.

Unavailable internals remain unavailable; failed fits retain errors without
replacement or invented K=1 results. Preflight failures or unavailable hierarchies
block the main gate. No truth-dependent branch or connectivity criterion enters
unsupervised fitting or automatic model/seed selection.

## CLI and API

Use Python 3.11/3.12. Install the optional neural extra for the recorded source
runtime; this diagnostic evaluates weights with NumPy and does not train or
import PyTorch. The archive audit verifies PyTorch package metadata as part of
the declared original runtime. Foundation configuration/help remain numerical-
dependency-free. Full scientific archival replay must use the source's recorded
numerical/Python/platform versions; synthetic installed smokes test engineering
compatibility across supported Python versions.

Choose a new output root with `approved-plan.md` copied there. Run:

```bash
super-deli diagnose-neural-moons bind --archive-root /absolute/neural-study --output-dir /absolute/new-diagnosis
super-deli diagnose-neural-moons freeze --archive-root /absolute/neural-study --output-dir /absolute/new-diagnosis --config examples/neural-moons/study.json
super-deli diagnose-neural-moons preflight --archive-root /absolute/neural-study --output-dir /absolute/new-diagnosis --workers 2
super-deli diagnose-neural-moons audit-preflight --archive-root /absolute/neural-study --output-dir /absolute/new-diagnosis
super-deli diagnose-neural-moons gate --archive-root /absolute/neural-study --output-dir /absolute/new-diagnosis --staging-bytes ACTUAL_MEASURED_BYTES
super-deli diagnose-neural-moons main --archive-root /absolute/neural-study --output-dir /absolute/new-diagnosis --workers 2
super-deli diagnose-neural-moons audit-main --archive-root /absolute/neural-study --output-dir /absolute/new-diagnosis
super-deli diagnose-neural-moons report --archive-root /absolute/neural-study --output-dir /absolute/new-diagnosis
```

API entry points are `neural_moons_archive.audit_archive`,
`neural_moons_study.freeze/run_phase/audit_phase/gate` and
`neural_moons_report.report` under `super_deli.benchmarking`. Output files are
exclusive; failed gates/attempts are preserved, not overwritten or resumed.
`archive-binding.json` contains the full private inventory; `frozen.json` binds
it to code, settings, runtime and protocol. Each observation retains its context,
reconstruction, fit arrays, results and checksums. Main reporting requires the
independent audit. `aggregate.json` contains only small retrospective summaries.

Require a measured 1 GiB new-retained-evidence cap, including staging/build/test
evidence, a 25% allowance plus 128 MiB reserve, at least 8 GiB free and at most
two single-thread workers. Dependency environments and regenerable tool caches
are accounted separately from scientific/build evidence. Project main fitting
time with a 25% allowance against one hour. Failed gates require an explicit
budget/scope amendment. Full weights/coordinates/trees and logs stay local;
only code/docs/invented fixtures/small summaries/figures are published.

The installed smoke uses `fixture=true` and an explicit invented source manifest
with fixed **untrained** coefficients, two tiny tables and no empirical inputs.
Its reduced counts are engineering tests, never scientific recovery results.
No source search, contact, empirical fitting, significance/stability rerun or
automatic follow-up improvement is included. Findings may remain mixed or
unresolved; that does not authorize a larger study.
