# Experimental unknown-K spectral clustering

The separate `cluster-tabular-spectral` CLI/API proposes a group count from a
complete continuous morphology **or** ecology table. It standardizes raw traits,
builds a neighbor graph and applies a prespecified eigengap heuristic. PCA, neural
training and generating labels are not used. This is a point-only research
workflow: no partition confidence, bootstrap stability, new p-value or calibrated
species-boundary inference is supplied. Existing models/defaults/schemas and
historical p-values remain unchanged.

At least 32 unique, explicitly independent biological units and two selected
nonconstant analytical continuous columns are required. The workflow rejects
missing, count and categorical features and repeated units. It neither imputes
values nor infers aggregation. Genomic, shape, grouped/integrative adapters and
unified-comparison schema changes are outside this checkpoint. A model cannot
verify independence or resolve a missing empirical sampling/reference gate.

## CLI and Python API

Use Python 3.11 or 3.12 and the existing optional numerical extra:

```bash
python -m pip install '.[clustering]'
super-deli cluster-tabular-spectral /absolute/path/dataset.json \
  --model-config /absolute/path/model.json \
  --modality traits --output-dir /absolute/path/new-spectral-run
```

The dataset uses the existing tabular ingestion schema. Declare one eligible
modality or select it explicitly. Change the example's feature names to match
your analytical columns. The new configuration requires an explicit graph
choice: five or fifteen other neighbors. There is no default selected from the
retrospective moons results. For example:

```json
{
  "schema_version": 1,
  "assume_independent_units": true,
  "feature_names": ["length", "width"],
  "n_neighbors": 15,
  "max_k": 6,
  "min_cluster_size": 5,
  "random_seed": 71323,
  "max_working_bytes": 268435456
}
```

```python
from pathlib import Path
from super_deli.core.schema import config_from_dict
from super_deli.io.config import read_json
from super_deli.models.spectral_config import SpectralConfig
from super_deli.workflows.tabular_spectral import run_tabular_spectral

dataset_path = Path("/absolute/path/dataset.json")
summary = run_tabular_spectral(
    config_from_dict(read_json(dataset_path)),
    model_config=SpectralConfig.from_dict(read_json("/absolute/path/model.json")),
    base_dir=dataset_path.parent,
    output_dir="/absolute/path/new-spectral-run",
    modality_name="traits",
)
```

The numeric API `fit_tabular_spectral(table, config=...)` returns the result
record and retained arrays. The model is transductive; there is no prediction
method for new units. Output directories must be new and are never overwritten.

## Fixed selection and abstention policy

Selected traits are centered and divided by population SD (ddof=0). Euclidean
graphs exclude self-neighbors, order exact distance ties by source identity and
use symmetric affinity `(B+B.T)/2`, with zero diagonal. The symmetric normalized
Laplacian's ascending eigenvalues give scores `lambda[K+1]-lambda[K]`.

The unique largest gap over K=1..max_k proposes K; it is numerically unresolved
when the largest gap or its lead over the runner-up is <=1e-8. A proposal at the
upper search boundary withholds the partition. No second-choice K is substituted.
The frozen validation uses max_k=6, not generating counts supplied to the model.

Connected K=1 produces a marked constant partition. Exactly K graph components
produce a marked direct-component partition; other disconnected cases withhold.
Connected K=2..max_k-1 uses ARPACK spectral embedding and guarded library QR
assignment. The embedding, graph, eigenvalues and assignment trace are retained.
Residual/Gram errors above 1e-8, projector error above 1e-6, QR ties at
1e-10*max(1,row maximum), a different label count, or groups below the configured
minimum size withhold the partition. Solver failures remain failures. No graph
reconnection, cluster merging, replacement fit or favorable-graph selection occurs.

Abstention guards identify numerical/search/support limitations; they are **not**
calibrated statistical uncertainty. An available K=1 is a heuristic result, not
acceptance of one population or proof of one species. An available K>1 is a
candidate partition, not a validated species delimitation. Noise and overlap can
make eigengap selection unreliable, and graph construction changes the question.
See [von Luxburg, section 8.3](https://people.eecs.berkeley.edu/~jordan/sail/readings/Luxburg07_tutorial.pdf)
and the [spectral estimator reference](https://scikit-learn.org/1.6/modules/generated/sklearn.cluster.SpectralClustering.html).

## Evidence and statuses

`summary.json` records available/withheld/failed status, proposed versus selected
K, exact reason, identities, configuration and source provenance. `point.npz`
retains raw/standardized values, scaling, graph, spectrum, gaps, actual embedding
and any partial numerical evidence. `assignments.csv` exists only for an
available partition; its arbitrary labels are zero-based. Withheld units are not
HDBSCAN noise and are never silently assigned to K=1. Input/workflow exceptions
retain `failure.json`; the CLI exits unsuccessfully on failures.

No existing significance workflow accepts this spectral family. Stability and
empirical validation remain separate future work. Compare availability and
failure bounds alongside recovery; good available-only agreement cannot erase
the withheld requests.

The separate fresh study is documented in [the protocol](unknown-k-spectral-study.md).
Its [completed results](unknown-k-spectral-results.md) show substantial Gaussian
overpartitioning and failed moons group-count recovery. This experimental rule
is not recommended for automatic species delimitation.
