# Matched-view signal diagnostic

This opt-in archival workflow traces the completed matched morphology–ecology
study. It does not train a network, fit a new GMM, select a model or change any
production default or p-value. Its inputs are synthetic observations from the
completed study, not an empirical biological cohort.

The fixed source set contains 60 main observations: morphology-only signal,
complementary signal and redundant signal, each at 64/256 matched individuals
and ten replicates. Six separately archived preflight observations are used for
resource and numerical checks. Labels and continuous generating coordinates
enter evaluation only after clustering.

Each observation has 21 fixed coordinate spaces: standardized raw views,
balanced concatenation, saved view PCs/fused profiles/joint PCs, saved separate
PCA/HDBSCAN scores, raw and standardized paired neural coordinates for all three
seeds, and standardized separate-view neural coordinates for all three seeds.
One unchanged EOM/root-allowed HDBSCAN is fitted in each space. The 1,260 main
and 126 preflight fits include 660 and 66 exact source-native replays. One earlier
archive compatibility probe used the same fixed density settings; its result is
retained separately and never used to choose a representation or setting.

The report separates distance geometry, linear signal recovery, decoder
reconstruction, compression, latent scaling, branch structure and final tree
selection. Invertible feature scaling changes distances but does not itself
delete information. Low in-sample linear recovery in a nonlinear map does not
prove that all information is absent. Four-group branch scores are post-fit
oracle diagnostics, never an admissible species or model-selection rule.

Five/fifteen-neighbor summaries exclude self and withhold boundary ties.
Noise is retained separately from clusters. Unavailable hierarchies and failed
fits retain reasons; no failed result becomes one cluster. Independent audits
check raw joins, frozen transformations and decoder arithmetic, dense graph
distances/MST, Python condensation, native tree selection, candidate GMM
likelihood/BIC and all source-native assignments. These checks are not a second
complete implementation of HDBSCAN.

The ten observations per cell are reused exploratory evidence. Neural seeds,
spaces and stages are repeated measurements. Exact count intervals are
descriptive and unadjusted; they do not establish a uniquely correct repair,
general performance, species probability or significance. The checkpoint ends
with one supported repair proposal, or an explicitly unresolved mechanism.

## Running the archival workflow

Install `super-deli[clustering]` under Python 3.11 or 3.12. PyTorch is not required
to replay saved numeric weights. The source root must contain the completed
matched study, its checksums/source bundle and completion receipt. Source paths
must stay within that root. Put a reviewed `approved-plan.md` in a separate new
output directory before binding and freezing.

```bash
super-deli diagnose-matched-view-signal bind --archive-root /absolute/archive --output-dir /absolute/new-output
super-deli diagnose-matched-view-signal freeze --archive-root /absolute/archive --output-dir /absolute/new-output
super-deli diagnose-matched-view-signal preflight --archive-root /absolute/archive --output-dir /absolute/new-output --workers 2
super-deli diagnose-matched-view-signal audit-preflight --archive-root /absolute/archive --output-dir /absolute/new-output
super-deli diagnose-matched-view-signal gate --archive-root /absolute/archive --output-dir /absolute/new-output --staging-bytes 123456
super-deli diagnose-matched-view-signal main --archive-root /absolute/archive --output-dir /absolute/new-output --workers 2
super-deli diagnose-matched-view-signal audit-main --archive-root /absolute/archive --output-dir /absolute/new-output
super-deli diagnose-matched-view-signal report --archive-root /absolute/archive --output-dir /absolute/new-output
```

Replace the example staging size with measured new local evidence bytes, including
logs and verification copies. The gate requires an independently audited
preflight, a 512 MiB retained-evidence cap, at least 8 GiB free, at most two
single-thread workers, and projected main execution plus audit under one hour.
Projection includes a 25% allowance and credits no parallel speedup. Freeze
source/runtime/settings before the preflight; do not alter them after outcomes.
Failed gates are retained and require a documented scope/budget amendment.

`examples/matched-view-signal/smoke.py` uses explicitly invented untrained
engineering fixtures. It tests the installed lifecycle, not scientific recovery.
Detailed inputs, weights, coordinates, labels and audit traces remain local;
only code, documentation, figures and small cell aggregates are published.
