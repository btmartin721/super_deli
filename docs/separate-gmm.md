# Separate encoders and a shared Gaussian mixture

`cluster-matched-separate-gmm` is a separately versioned, opt-in workflow for
complete continuous morphology and individually measured ecology from the same
independent individuals. Explicit specimen IDs align the views. The original
separate-encoder HDBSCAN workflow, configurations and p-values are unchanged.

Each view uses its existing preprocessing and two-coordinate CPU autoencoder.
Four standardized coordinates are concatenated without another reduction. A
shared-covariance Gaussian mixture chooses K by BIC, including K=1. Defaults are
K=1..4, minimum hard size five, ten starts, 500 EM iterations, tolerance 0.001 and
regularization 1e-6. Neural seed and mixture seed are separate declarations.

```bash
super-deli cluster-matched-separate-gmm dataset.json \
  --model-config shared-model.json --output-dir /absolute/path/to/new-output
```

A runnable invented example uses the existing separate-neural dataset:

```bash
super-deli cluster-matched-separate-gmm examples/separate-neural/dataset.json \
  --model-config examples/separate-gmm/model.json --output-dir /absolute/path/to/new-example
```

Use Python 3.11/3.12 and `super-deli[neural]`. The model JSON follows the separate
encoder contract, with schema 1 and family
`matched_separate_autoencoders_shared_gmm`. Additional settings are `max_k`,
`min_cluster_size`, `n_init`, `max_iter`, `tol`, `reg_covar` and `mixture_seed`.
Each view requires 3–32 nonconstant selected continuous traits. Missingness,
unmatched/grouped individuals, shared locality ecology, genomic and shape inputs
are excluded from this adapter. Declaring independent individuals does not verify
sampling independence or remove confounding/allometry.

Outputs retain both branch networks/preprocessing/losses, four-coordinate scores,
all selected-component memberships, candidate criteria/eligibility and the fitted
shared covariance. `assignments.csv` names conditional mixture memberships
explicitly; they are not density strengths or species probabilities. CSV cluster
IDs start at zero: `component_1_probability` corresponds to cluster 0, displayed
as `cluster_1` in the HTML report. Both report
panels show only one branch each; clustering uses all four coordinates. A K at the
configured maximum is flagged as a search boundary, not proof of the true count.

`load_separate_gmm_model(path)` provides frozen `encode(tables)`,
`reconstruct(tables)` and conditional `predict(tables)` operations without
importing Torch or SNPio. `predict` returns IDs, scores, labels and membership.
These projections hold all parameters fixed; they provide no external predictive
validation or out-of-distribution guarantee. Numeric models use checked NPZ
arrays without pickle. New analyses never replace existing directories.

Optional `n_subsamples` uses the existing floor(0.8*N) whole-individual sampling
without replacement. Both encoders and the entire mixture search refit on each
matched subset. Inclusion/coassignment arrays retain failures and denominators;
absent units receive no invented labels. Neural initialization is fixed within
draws. This is conditional sensitivity, not a species probability or p-value.
A numerical candidate failure withholds the search; it never becomes a K=1 result.

The [fresh study protocol](separate-gmm-study.md) evaluates recovery and splitting
of Gaussian controls separately. Gaussian components can describe distributional
shape rather than biological species. No significance calibration is added and
no family, seed or new default is automatically chosen.

This adapter retains the existing conservative paired-workflow memory limits;
large-cohort scaling is not evaluated by the N=64/256 study. Model memberships
remain conditional even when K=1 gives every individual membership one.
