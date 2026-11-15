# Bounded multimodal signal-loss diagnosis

This benchmark-only command investigates signal lost when a matched genomic,
morphological, ecological and fixed-landmark analysis is compressed into a
combined representation. It does not change analysis defaults, fit species
boundaries or calculate p-values. The existing public analysis interface is
[matched multimodal discovery](matched-multimodal.md).

## Scope and reproducibility

The source is the completed version 0.81.0.dev0 study at commit
`5936eab934772e08e40d275d3d492389b9696e0f`. Supply the SHA-256 seals for its
`completion.json` and `study-manifest.json` in a `SignalLossConfig` schema-1 JSON.
The reader verifies bounded regular files against that inventory. Invented
contract fixtures are explicitly marked and cannot authorize scientific runs.

```bash
super-deli diagnose-multimodal-signal config.json \
  --source /path/to/completed/matched-multimodal-study \
  --output-dir /path/to/new/archived-audit --phase audit

super-deli diagnose-multimodal-signal config.json \
  --source /path/to/completed/matched-multimodal-study \
  --output-dir /path/to/new/study --phase preflight \
  --generator examples/multimodal/synthetic.py

super-deli diagnose-multimodal-signal config.json \
  --source /path/to/completed/matched-multimodal-study \
  --output-dir /path/to/new/study --phase main \
  --generator examples/multimodal/synthetic.py
```

The archive audit is fit-free. Scientific phases require the sealed production
source and a frozen local invented-data generator. Preflight must pass numerical,
replay, storage and runtime guards before the main run. Output directories must
be new; existing results are not overwritten. The command is a bounded study
protocol rather than a general-purpose configuration search.

## Fixed comparisons

The diagnostic traces original unit IDs through raw inputs, per-view PCA,
view normalization, fusion and joint PCA. It replays three native workflows per
observation and retains every attempted mixture candidate. Read-only observers
leave the estimator and its original failure policy unchanged.

The cached-score comparisons reuse the native training-view transformations:

- All four views with two, four or eight joint PCs, using a minimum cluster
  size of nine throughout this contrast. This preserves the existing rule that
  the minimum exceed the number of fitted coordinates. The native two-PC,
  minimum-five baseline remains separately identified.
- Five fixed pairs, adding each nonsignal view to shape or genomic data, with
  two joint PCs and minimum five.
- Genomic diagonal/shared and shape full/shared covariance heads, on a common
  explicit row order. Native standalone replays retain their original order.

Settings are fixed before fresh generation: equal view weights, K=1 through 4,
ten EM starts and unchanged model seeds. Truth labels enter evaluation only.
Likelihood gains and BIC penalties are compared within each score space;
absolute BIC values across different dimensions are not ranked.

## Validation and interpretation

The main study contains 24 archived and 48 fresh observations: four fresh per
law/sample-size cell under continuous controls, redundant signal, complementary
signal, conflicting partitions, genomic-only signal and shape-only signal, at
N=64/256. It requests 1,080 searches. A separate 72-search preflight includes
exact archived native replays. There is no bootstrap, significance calibration,
new neural training or empirical fitting.

Signal retention measures how much simulated group-centroid separation survives
a projection. It is not a species probability or an information-theoretic bound.
In-sample linear recovery can overfit in high dimension. Continuous controls and
conflicting laws have no privileged four-species truth. Four fresh observations
per cell support only a descriptive operating-range check.

The frozen follow-up gate requires archived shape losses, numerical full-coordinate
retention, at least two additional fresh shape recoveries, no additional control
splits, no complementary recovery loss and no additional failures. A passing gate
would authorize recommending a bounded opt-in follow-up, not a default change.

Detailed source rows, fitted parameters, masks, labels and candidate arrays stay
local. Only code, invented tests and small aggregates are published. See the
[completed results](multimodal-signal-results.md).
