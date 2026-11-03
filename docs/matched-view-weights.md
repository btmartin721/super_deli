# Prespecified matched neural view weights

`cluster-matched-neural` accepts an opt-in schema-2 model configuration with
`view_loss_weights: [0.8, 0.2]` for morphology/ecology, respectively. Both
fractions must be finite and strictly positive and sum to one. Declare them
before fitting; the workflow does not choose weights or a winning neural seed.
See `examples/matched-view-weights/model.json`.

This workflow requires complete continuous traits from the same independently
sampled individuals, explicit identity links and individually measured ecology.
It does not copy shared locality ecology onto individuals as independent data.
After standardization, each retained feature in view v is multiplied by
`sqrt(D * w_v / d_v)`, where D is the total retained feature count and d_v is
the view's retained count. This changes both encoder inputs and reconstruction
emphasis. The architecture, 500 CPU training updates, latent standardization,
HDBSCAN rules and noise handling remain unchanged.

Schema 1 retains its existing serialized fields and arithmetic. Schema-2 50/50
uses the same numerical path, adding weight provenance. Saved models retain
declared fractions, normalized fractions and effective feature scales. Frozen
encoding/reconstruction needs NumPy, not PyTorch; it supplies no density labels
for new individuals. Native 80% unit subsampling refits all preprocessing within
each draw and retains paired source identities. No new p-values are computed.

## Bounded simulation study

The separate `benchmark-view-weights` CLI exposes `bind`, `freeze`, `preflight`,
`audit-preflight`, `gate`, `main`, `audit-main` and `report`. Every action requires
`--archive-root` and `--output-dir`. Use a new local output directory containing
the approved plan as `approved-plan.md`; bind the original completed matched
study for twelve legacy replays. This archival benchmark is optional and is not
required to run weighted clustering. Existing data, models and evidence are
never overwritten.

The frozen design uses 60 fresh observations, six signal/control laws and
N=64/256, retaining all three neural seeds and three fractions (50/50, 80/20,
20/80). Nine joint arms, six single-view neural references and the unchanged
integrative PCA/GMM baseline give 960 point workflows. Add 108 matched native
refits and twelve exact archival replays: 1,080 main workflows. The separately
audited preflight contains 300 workflows. Generating labels are evaluated after
fitting and never enter preprocessing, training or weight selection.

Read-only NumPy checks inspect actual gradients, moments and Adam updates during
primary training, with independent loss evaluation and source/identity/transform,
density-hierarchy and GMM likelihood audits. No second neural optimization is
performed for each scientific fit. Partial failures remain failures, with no
replacement observations or K=1 fallback. Numerical/resource guards stop
dependent work and require an explicit amendment.

The resource gate caps detailed local evidence at 2 GiB including verification
staging, requires 8 GiB free, allows at most two single-thread workers and
projects main execution plus audit under one hour with a 25% allowance and no
parallel speed credit. Detailed inputs, models and diagnostics remain local.
Only code, documentation and small aggregates are published.

Reports distinguish four-group recovery (K=4 and full ARI >=0.8), Gaussian-control
splitting, noise/coverage, failures, per-view reconstruction, linear signal
retention and illustrative native stability. The 108 primary rates and 72 paired
contrasts have marginal and simultaneous intervals. Five independent datasets
per cell give descriptive sensitivity, not general superiority; three subsamples
cannot establish general stability. Seeds and weights are repeated measurements.
Different weighted losses optimize different objectives and are not ranked.
Control splitting is not a calibrated type-I error and clusters are not validated
species. Defaults, historical p-values and empirical readiness gates are preserved.
