# Opt-in denoising autoencoder and density groups

This separate research workflow learns two neural coordinates from one complete
continuous morphology or ecology table, then runs the existing HDBSCAN density
model. It leaves ordinary neural, PCA/GMM, PCA/HDBSCAN and significance defaults
unchanged. Python 3.11 and 3.12 are supported.

Install the optional CPU neural dependencies in your environment:

```bash
python -m pip install 'super-deli[neural]'
```

Use the same [typed dataset declaration](tabular-neural.md) as ordinary neural
analysis. Select three to 64 continuous analytical features, at least three
nonconstant, with at least ten independent biological units and complete finite
values. Declare independence explicitly; the software cannot verify it.

Write a new model configuration, replacing the illustrative feature names with
columns in your dataset:

```json
{
  "schema_version": 1,
  "model_family": "tabular_denoising_autoencoder_hdbscan",
  "feature_names": ["length", "width", "depth"],
  "assume_independent_units": true,
  "random_seed": 0,
  "corruption_seed": 101,
  "subsampling_seed": 202,
  "n_subsamples": 20
}
```

```bash
super-deli cluster-tabular-denoising dataset.json \
  --model-config denoising.json --output-dir denoising-run
```

The output directory must be new. Source tables are preserved. Read
`assignments.csv`, `report.html`, `summary.json`, `training.json` and
`subsamples.json` together. Cluster -1 means noise/unassigned; it is not one
shared group. `point.npz` contains numeric weights, scales, raw/standardized
training encodings, clean reconstruction losses, corrupted training objectives,
four saved weight checkpoints and all 500 corruption digests. `stability.npz`
retains refits and inclusion-conditional coassignment with failure bounds.
Failed points and draws retain available evidence and are not replaced.

## What changes during training

The fixed network and Adam settings match the ordinary autoencoder: encoder
d -> 16 -> 8 -> 2, mirrored decoder, tanh hidden layers, float64 CPU, 500 full-batch
updates, learning rate 0.001 and weight decay 0.0001. At each update, independent
Gaussian corruption with SD 0.15 is added to the standardized input. The target
is the original **observed** standardized measurement, not an unknown true value.
No corruption is added during encoding or reconstruction after training.

A separate local CPU generator makes corruption reproducible without changing
process RNG state. The training recipe is fixed for version one; architecture,
noise-level searches, early stopping and best-seed selection are not provided.
Latent axes receive fit-unit centering/population-SD scaling before unchanged
HDBSCAN: minimum size five, min_samples five including self, EOM with one cluster
allowed. Better reconstruction need not improve clustering.

The method is motivated by [Vincent et al. (2008)](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf).
Their results do not establish species-delimitation accuracy for this model.

## API and frozen numeric reuse

```python
from super_deli.io.config import load_config
from super_deli.models.denoising_config import DenoisingConfig
from super_deli.workflows.tabular_denoising import (
    load_denoising_model,
    run_tabular_denoising,
)

dataset = load_config("dataset.json")
settings = DenoisingConfig(
    feature_names=("length", "width", "depth"),
    assume_independent_units=True,
    corruption_seed=101,
    n_subsamples=20,
)
summary = run_tabular_denoising(
    dataset, model_config=settings, base_dir=".", output_dir="denoising-run"
)
model = load_denoising_model("denoising-run")
# model.encode(values) and model.reconstruct(values) require complete rows in
# the exact declared feature order. Numeric reload does not require PyTorch.
```

There is no density-label prediction for new or absent units. The fit is
transductive; reconstruction is training performance, not held-out validation.
Whole-unit 80% subsampling refits preprocessing, training and density groups,
conditional on the chosen initialization/corruption seeds. Absent units receive
no invented labels; failed draws remain in failure-bound denominators. Changing
row count/order changes application of the reproducible corruption stream.
Stability and density strengths are not species probabilities or p-values.

Missing data, counts, categorical traits, grouped or integrative data, genomes
and shapes are outside this adapter. Empirical sampling/reference gates remain
unchanged. No neural significance calibration or species-boundary guarantee is
provided. See the [frozen comparison](denoising-study.md) and
[completed findings](denoising-results.md).
