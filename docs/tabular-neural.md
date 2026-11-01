# Tabular autoencoder and HDBSCAN

The opt-in `cluster-tabular-neural` CLI/API is Super Deli's first neural model.
It learns a nonlinear two-dimensional representation of one complete continuous
morphology or ecology table, then applies HDBSCAN with explicit noise assignments.
Existing models, defaults, schemas, p-values and empirical evidence gates retain
their behavior. This workflow does not identify validated species boundaries.

## Install and run

Use Python 3.11 or 3.12. Install `super-deli[neural]`; PyTorch is not required by
the foundation or existing classical workflows. For a CPU-only Linux/WSL setup,
install PyTorch from its CPU index first:

```bash
python -m pip install 'torch==2.15.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[neural]'
python examples/neural/generate_example.py /tmp/new-neural-example
super-deli cluster-tabular-neural /tmp/new-neural-example/raw/dataset.json \
  --model-config /tmp/new-neural-example/model.json \
  --modality traits --output-dir /tmp/new-neural-fit
```

The example is invented software-check data, not empirical morphology or species
truth. Use new output directories. Dataset paths resolve relative to the dataset
JSON. Sources are staged and fingerprinted, without modifying original files.

## Input and configuration

Select exactly one numeric morphology or ecology modality. Each row must have a
unique observation and biological-unit ID, with an explicit independence
assumption. Specimen, population and locality rows retain their declared level.
This workflow does not aggregate specimens or combine different modalities.

Select 3–64 continuous analytical features. Counts, categories, covariates as
predictors, missing cells and repeated units are rejected. Constant features are
recorded and excluded; at least three must remain. Each fit requires at least
ten units. An 80% subsample can fail that minimum even if its original fit is
allowed. No data are manufactured to rescue an unsupported fit.

```json
{
  "schema_version": 1,
  "model_family": "tabular_autoencoder_hdbscan",
  "assume_independent_units": true,
  "feature_names": ["trait_1", "trait_2", "trait_3", "trait_4"],
  "random_seed": 0,
  "subsampling_seed": 0,
  "n_subsamples": 0
}
```

The feature list, schema version and independence assumption are required.
Unknown fields and unsupported versions fail. Optional workspace and pairwise
budgets default to 256 MiB each. The independence declaration is an analyst's
assumption, not a check for relatives, shared sampling events or pseudoreplication.

## Fixed first neural model

The encoder has widths `d -> 16 -> 8 -> 2` and the decoder `2 -> 8 -> 16 -> d`.
Hidden layers use tanh, with linear bottleneck and output layers. Fit-unit means
and population SDs standardize raw features. No PCA precedes the encoder.

Training uses float64 CPU tensors, full-batch Adam, learning rate 0.001, weight
decay 0.0001 and exactly 500 updates. The retained loss history contains the
initial MSE and all 500 post-update MSEs. There is no early stopping, supervised
label input or winning-seed search. Finite training loss is not a convergence or
biological-validity guarantee. An autoencoder learns by reconstructing its
inputs; a useful reconstruction need not preserve a desired cluster structure.
See the foundational [autoencoder paper](https://www.cs.toronto.edu/~hinton/absps/science.pdf).

The fitted two latent axes are standardized using fit-unit means and population
SDs before Euclidean HDBSCAN. This metric choice is recorded. A comparison with
PCA/HDBSCAN evaluates whole workflows rather than isolating only the encoder.
HDBSCAN uses minimum cluster size five, min_samples five, EOM selection,
allow_single_cluster true, alpha one and epsilon zero. It has no prescribed K.
Label -1 means noise/unassigned, including a valid all-noise result with K=0.

The network is trained on the analysis cohort. These are **transductive** groups:
the procedure describes those observations. Training reconstruction is not
held-out prediction evidence. IDs, origin labels, generating truth and other
reference classifications never enter the feature matrix or model selection.
The saved encoder can transform new measurements, but the workflow provides no
HDBSCAN species assignment or density strength for previously unseen units.

## Conditional subsampling

Setting `n_subsamples` above zero requests independent 80% whole-unit subsamples
without replacement. Each draw refits raw scaling, network weights, latent
scaling and HDBSCAN. The original neural initialization seed is held fixed;
`subsampling_seed` controls the separate unit draws. This separates the sampling
perturbation from an additional randomized initialization choice.

Only included units receive draw labels. Noise observations never form a shared
cluster. Coassignment divides positive-cluster matches by successful pair
inclusions. Inclusion counts, noise frequencies, K frequencies, conditional
Monte Carlo SEs and failed-outcome bounds remain explicit. Never-included pairs
are unavailable, not zero. Failed draws retain identities, errors and available
training checkpoints, without replacement draws. Incomplete results remain
marked incomplete even when some conditional frequencies can be calculated.

These measures are conditional sensitivity statistics. They are not species
probabilities, biological sampling confidence intervals or significance levels.
The separate study reports sensitivity to neural initialization; it is not
treated as additional biological replication or pooled with sampling stability.

## Outputs and API

- `assignments.csv`: exact IDs, native density labels, strengths and latent scores.
- `report.html`: one offline report with a latent-space plot and assignment table.
- `point.npz`: numeric weights, raw/latent transforms, scores, labels and loss history.
- `training.json`: fixed architecture/training, numerical diagnostics and exclusions.
- `subsamples.json`, `stability.npz`: draw identities, failures, fitted arrays and
  inclusion-conditional support with explicit denominators.
- `summary.json`: completion, configuration/source/artifact hashes, runtime versions,
  assumptions and explicitly uncomputed species significance.
- `failure.json`, and available `failed-training.npz`: retained failed-point evidence.

Weights are numeric arrays read with `allow_pickle=False`, not serialized Python
estimators. A frozen model can be loaded and evaluated using NumPy without
importing PyTorch:

```python
from super_deli import load_config
from super_deli.models.neural_config import NeuralConfig
from super_deli.workflows.tabular_neural import run_tabular_neural, load_neural_model

source = "/absolute/path/to/dataset.json"
configuration = NeuralConfig(("trait_1", "trait_2", "trait_3", "trait_4"), True)
summary = run_tabular_neural(
    load_config(source), model_config=configuration,
    base_dir="/absolute/path/to", output_dir="/absolute/path/to/new-neural-run",
    modality_name="traits",
)
model = load_neural_model("/absolute/path/to/new-neural-run")
# new_values must be complete numeric rows in exactly the declared feature order.
# model.encode(new_values) returns frozen standardized latent coordinates.
# model.reconstruct(new_values) returns reconstructed selected raw features.
```

The CLI returns nonzero for a failed point or incomplete requested subsampling.
Existing outputs are never overwritten. Global CPU RNG and thread/determinism
settings are restored. Reproducibility is checked within a recorded runtime;
identical results across releases, devices or platforms are not promised. See
[PyTorch's reproducibility notes](https://docs.pytorch.org/docs/2.15/notes/randomness.html).

## Scope and validation

The [bounded simulation protocol](tabular-neural-study.md) compares this model
with unchanged PCA/GMM and PCA/HDBSCAN. It includes Gaussian controls, four groups,
a nonlinear moons embedding and a continuous cline at N=64/256, with fixed
settings and retained failures. See the [executed results](tabular-neural-results.md).

No existing GMM calibration is transferred to the neural model. Missing values,
genomic/shape/multimodal neural adapters, anatomical landmark learning, grouped
sampling designs and automatic model/default selection remain later work.
