# Unified matched morphology/ecology discovery comparison

Use complete continuous measurements from the **same individuals**, with explicit
specimen IDs and individually measured ecology. This opt-in coordinator invokes
the existing PCA/shared-GMM and separate-encoder/shared-GMM workflows. It changes
no numerical estimator, old configuration, default or historical p-value.

```bash
python -m pip install '.[neural]'
super-deli analyze-matched-tabular examples/matched-analysis/dataset.json \
  --analysis-config examples/matched-analysis/analysis.json \
  --output-dir /absolute/path/to/new-matched-comparison
```

Use Python 3.11 or 3.12 and a new output directory. With only PCA arms, the
`clustering` extra suffices; configuration parsing and CLI help do not import
Torch. The bundled inputs are invented 32-specimen measurements for a software
example, not an empirical validation dataset.

```python
from pathlib import Path
from super_deli import load_config
from super_deli.io.config import read_json
from super_deli.models.matched_analysis_config import MatchedAnalysisConfig
from super_deli.workflows.matched_analysis import run_matched_analysis

source = Path("examples/matched-analysis/dataset.json").resolve()
summary = run_matched_analysis(
    load_config(source),
    analysis_config=MatchedAnalysisConfig.from_dict(
        read_json(source.parent / "analysis.json")
    ),
    base_dir=source.parent,
    output_dir="/absolute/path/to/new-matched-api-run",
)
```

## Explicit native choices

The schema-1 analysis requires `ecology_measurement_scope: "individual"` and one
to sixteen uniquely named arms. Each arm has exactly `name`, `family` and `model`.
Use `pca_gmm` with an existing `IntegrativeConfig`, or `separate_gmm` with an
existing `SeparateGMMConfig`. Full native model configurations are retained.

PCA arms require two ordered morphology/ecology views, specimen targets, shared
(`tied`) covariance, BIC, standard scaling, zero assignment threshold, no
aggregation, complete views and one numerical thread. Neural arms use their
existing contract. All arms must select identical modality names and ordered
features. They may declare different native dimensions, seeds, searches and
resampling settings. These differences are visible, not silently harmonized.
The common input contract selects 3–32 continuous features per view and at least
ten individuals, reusing existing matched validation and memory limits.

Every declared specimen needs exactly one observation in both selected views.
Source row order and observation names do not imply matching. Missing cells,
unequal coverage, count/categorical analytical features, grouped/unmatched data,
shared locality ecology and ecological broadcasting are rejected. Unselected
modalities remain in the original manifest but do not enter numerical fitting.
Declaring independence and individual ecology does not verify their truth.

## Outputs and interpretation

`inputs/` contains checksum-verified source copies, the selected dataset and
fit-free cohort/row maps. `arms/<name>/` retains each native workflow's parameters,
assignments, component search, preprocessing, resampling records and failures.
The original input files are unchanged. Each arm is checked against the frozen
cohort, source identities, configuration and native model fingerprint.

Open `report.html` locally. It contains the prespecified choices, candidate K,
search-boundary flags, label-invariant partition agreement, native resampling and
links to the authoritative native artifacts. Its two-dimensional displays show
joint PCs for PCA and the morphology branch for the neural model; clustering
uses every retained coordinate. Colors identify components within one arm and
are not a cross-model component correspondence.

`assignments.csv` joins all arms by canonical specimen ID and retains both source
observation IDs. Arm-specific cluster columns use zero-based native labels;
maximum conditional mixture memberships are separate columns. The same numeric
label in two arms does not establish a match; compare specimen membership and
label-invariant agreement. A failed arm has
blank assignments, never invented K=1. `models.csv`, `comparisons.csv/JSON` and
`summary.json` preserve completion/partial/failure states and unavailable metrics.
CLI exits 1 for a partial or failed comparison. A valid point can be retained
when its requested resampling is incomplete; native failure counts remain explicit.

Agreement compares partitions, not biological accuracy. BIC is used inside each
native fixed-space K search; absolute BIC values are not ranked across different
representations. No winner, consensus species partition or feature weight is
selected from the comparison.

Conditional mixture membership, PCA whole-unit bootstrap, neural matched 80%
subsampling and significance answer different questions. Native resampling keeps
its own draw multiplicities, inclusion/OOB coverage, failures and denominators.
The new report never pools supports. `inference.status` is `not_computed` with no
p-value. Existing standalone Gaussian-departure calibration commands remain
separate and do not establish the selected K or species boundaries.

This checkpoint runs no empirical cohort or author contact and does not relax
empirical sampling/reference gates. See [the integration validation](matched-analysis-results.md).
