# Matched morphology–ecology neural integration

`cluster-matched-neural` learns a shared two-dimensional representation of
complete continuous measurements from the same individuals, then applies the
existing HDBSCAN rule. Its view-balanced reconstruction loss gives morphology
and ecology equal loss weight despite different numbers of selected traits.
This does not guarantee equal biological influence or improved group recovery.
The workflow is opt-in, transductive and experimental; it supplies candidate
groups and conditional subsampling summaries, not validated species or p-values.

## Run the invented example

Use Python 3.11 or 3.12 and the optional `neural` dependencies. For Linux/WSL,
the repository's CPU setup is:

```bash
python -m pip install 'torch==2.15.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[neural]'
python examples/matched-neural/generate_example.py /tmp/new-matched-example
super-deli cluster-matched-neural /tmp/new-matched-example/raw/dataset.json \
  --model-config /tmp/new-matched-example/model.json \
  --output-dir /tmp/new-matched-results
```

Use new output directories. The example is synthetic, with deliberately
shuffled ecology rows to demonstrate ID alignment. It is not an empirical
species-validation dataset. Paths in dataset JSON resolve relative to that file.

## Input contract

Select one morphology and one ecology modality at **specimen level**, with one
observation per identical set of unique individual IDs. The reader retains both
observation-ID lists and source row indices; fitting/output order is sorted unit
ID. It never joins by row position or a reference label, takes an implicit
intersection, averages specimens, or copies locality ecology onto individuals.

Declare independent individuals and individually measured ecology explicitly.
Those are analyst assumptions, not verification of sampling or species truth.
Selected traits must be continuous analytical features: 3–32 per view, at most
64 total, with every selected cell finite and observed. The fit needs at least
ten individuals. Constants are recorded/excluded within each fit; each view
must retain at least three nonconstant traits, including in subsamples.
Duplicate, repeated, missing, unmatched, mixed-level or incorrectly typed inputs
fail clearly. Other dataset modalities may remain unselected and unread.

```json
{
  "schema_version": 1,
  "model_family": "matched_morphology_ecology_autoencoder_hdbscan",
  "morphology": "morphology",
  "ecology": "ecology",
  "morphology_features": ["m1", "m2", "m3", "m4"],
  "ecology_features": ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"],
  "assume_independent_units": true,
  "ecology_measurement_scope": "individual",
  "random_seed": 0,
  "subsampling_seed": 0,
  "n_subsamples": 0
}
```

Morphology/ecology feature names are scoped to their respective modality; the
same name can occur in both. Frozen transforms require the same definitions
and measurement units. No automatic unit conversion is performed.

## Fitting and uncertainty

Fit-unit means/population SDs standardize each selected trait. For p_M,p_E
retained traits and D=p_M+p_E, multiply the standardized views by
sqrt(D/(2*p_M)) and sqrt(D/(2*p_E)) **after** standardization. The unchanged
ordinary autoencoder trains on this concatenation, so its MSE equals the mean
of the two per-view standardized reconstruction MSEs. Weighting before another
standardization would cancel the balance and is not done here.

Reuse the fixed D->16->8->2 encoder and reverse decoder, tanh hidden layers,
float64 CPU full-batch Adam and 500 updates from the single-view workflow.
There is no PCA, denoising, early stopping or winning-seed search. Standardize
latent coordinates before existing HDBSCAN: minimum cluster size five,
min_samples five including self, EOM, root allowed, alpha one and epsilon zero.
K is not prescribed. Noise is -1; all-noise K=0 is a valid density result.
A finite training loss is not a convergence or biological-validity guarantee.

The default computes no stability. A positive `n_subsamples` requests native
80% whole-individual subsampling without replacement. Both views use the same
individuals, and every draw refits view means/scales/constants/weights, network,
latent scaling and density groups. Initialization stays fixed; the independent
subsampling seed selects individuals. Failed draws remain failed and are not
replaced. A valid full fit can have unsupported smaller subsamples.

Coassignment uses successful pair inclusions; noise never forms a shared group.
Pair-inclusion counts, noise/coverage, K frequencies and failed-outcome bounds
remain separate. Unseen pairs are unavailable, not zero. These conditional
frequencies and Monte Carlo summaries are not calibrated species supports.
No significance test is added, and GMM p-values cannot be transferred here.

## Artifacts and Python API

The new directory contains staged source ingestion, `cohort.json` with both
observation crosswalks, `assignments.csv`, `point.npz`, `training.json`,
`stability.npz`, `subsamples.json`, `summary.json` and one local `report.html`.
Network weights are numeric, without pickle. Sources and configurations are
fingerprinted; failures retain diagnostics and any partial numeric evidence.
A failed output does not retain a valid `assignments.csv` filename.

```python
from super_deli import load_config
from super_deli.models.matched_neural_config import MatchedNeuralConfig
from super_deli.io.config import read_json
from super_deli.io.tabular import ingest_tabular
from super_deli.workflows.matched_neural import (
    run_matched_neural, load_matched_neural_model,
)

# Use absolute paths appropriate to your dataset.
dataset = load_config('/absolute/path/dataset.json')
config = MatchedNeuralConfig.from_dict(read_json('/absolute/path/model.json'))
summary = run_matched_neural(
    dataset, model_config=config, base_dir='/absolute/path',
    output_dir='/absolute/path/new-results',
)
model = load_matched_neural_model('/absolute/path/new-results')
new_data = ingest_tabular(
    load_config('/absolute/path/new/dataset.json'), base_dir='/absolute/path/new',
    output_dir='/absolute/path/new-ingestion',
    modality_names=[config.morphology, config.ecology],
)
encoded = model.encode(new_data.tables)       # unit_ids and coordinates
reconstructed = model.reconstruct(new_data.tables)  # unit_ids and nested views
# reconstructed["views"][config.morphology] contains raw-scale reconstructions.
```

Reload validates hashes, feature/view dimensions, weights and numeric model
shapes. Frozen encoding/reconstruction uses NumPy without requiring PyTorch.
It does not refit means/weights and does not predict HDBSCAN group labels,
density strength or species membership for new individuals. Original empirical
sampling/reference gates remain closed. Genomic/shape neural inputs, missing
values, grouped/locality ecology and unmatched integration are separate work.

The [frozen validation protocol](matched-neural-study.md) specifies the paired
comparisons, all seeds, independent optimizer audit and resource limits.
The [completed pilot results](matched-neural-results.md) document complementary
signal recovery alongside uninformative-view losses, initialization differences,
limited small-sample stability and weak moons recovery. No setting is promoted.
