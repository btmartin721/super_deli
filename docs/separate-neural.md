# Separate morphology/ecology encoders

`cluster-matched-separate-neural` is an opt-in CLI/API for complete continuous
measurements from the same independent individuals. It fits the unchanged CPU
autoencoder separately to each view, retains two standardized coordinates per
view, and applies HDBSCAN to their four-coordinate concatenation. There is no
joint PCA, second compression, learned weight or automatic seed selection.
The existing joint encoders, defaults, schema-1/schema-2 configurations and
historical p-values retain their behavior. Use Python 3.11/3.12 and the optional
`neural` dependencies.

```bash
super-deli cluster-matched-separate-neural dataset.json \
  --model-config separate-model.json --output-dir /absolute/path/to/new-results
```

The model JSON uses schema 1 and family
`matched_separate_autoencoders_hdbscan`; declare morphology/ecology names, selected
feature lists, `assume_independent_units: true` and
`ecology_measurement_scope: "individual"`. See the invented example under
`examples/separate-neural/`. Both views must contain exactly the same unique
declared IDs; source row order supplies no correspondence. The workflow rejects
missing, repeated, unmatched or mixed-level rows and noncontinuous features.
It does not verify independence, adjust allometry, select empirical cohorts or
copy shared locality ecology onto individuals.

Each view requires 3–32 nonconstant selected traits and at least ten matched
individuals. Both branches use the declared neural seed independently, the fixed
d/16/8/2 encoder, inverse decoder, float64/tanh, existing Adam settings and 500
updates. Two encoders therefore perform 1,000 updates and have different parameter
counts from a single joint network. Equal concatenated view influence does not
establish equal biological importance. Joint rank two through four is recorded;
genuine redundancy is retained without whitening or variance floors. Existing
per-encoder collapse checks remain in force.

Numeric artifacts retain both preprocessors, networks, all losses, latent
normalization, identities, four-coordinate scores and density strengths. The
configuration/source/model fingerprints protect provenance. The HTML report
shows separate branch panels colored by the joint assignments; neither panel
depicts the full four-dimensional clustering geometry. Per-view reconstruction
and their descriptive average are training summaries, not a joint optimization
objective or external predictive validation.

`load_separate_neural_model` supports frozen `encode(tables)` and
`reconstruct(tables)` without importing PyTorch or SNPio. Future units receive
coordinates/reconstruction, not invented HDBSCAN labels. New outputs never
replace existing directories. Failures retain partial evidence and no completed
assignments.

Optional `n_subsamples` requests native 80% whole-unit subsampling without
replacement. Both branches refit on exactly the same selected identities in each
draw, with all preprocessing/normalization refitted and neural seeds fixed.
Support arrays distinguish attempted/successful inclusions, noise and failures;
absent units receive no inferred labels. This is conditional sensitivity, not
species probability or a p-value. Defaults request zero draws.
