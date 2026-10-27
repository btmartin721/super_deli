# Unified grouped morphology/ecology analysis

`analyze-grouped-tabular` compares prespecified existing grouped GMM workflows on
one immutable cohort. It supports full/shared covariance, ordinary BIC/AIC and
predictive-screened AIC, with one offline report and the native evidence intact.

Run from the repository root:

```bash
python -m pip install -e '.[clustering]'
super-deli analyze-grouped-tabular examples/grouped-analysis/dataset.json \
  --analysis-config examples/grouped-analysis/analysis.json \
  --output-dir /tmp/super-deli-grouped-comparison
```

Use a new output directory. Dataset input paths resolve relative to the dataset
JSON. The public fixture contains invented identities and simulated traits.
The separate `analyze-tabular` command and its schema-1 configuration retain
their original scope and behavior.

```python
from pathlib import Path
from super_deli import load_config
from super_deli.io.config import read_json
from super_deli.models.grouped_analysis_config import GroupedAnalysisConfig
from super_deli.workflows.grouped_analysis import run_grouped_tabular_analysis

source = Path("examples/grouped-analysis/dataset.json").resolve()
summary = run_grouped_tabular_analysis(
    load_config(source),
    analysis_config=GroupedAnalysisConfig.from_dict(
        read_json(source.parent / "analysis.json")
    ),
    base_dir=source.parent,
    output_dir="/tmp/super-deli-grouped-python",
)
```

## Complete, explicitly grouped inputs

Select exactly two views in morphology/ecology order: one complete continuous
morphology record per declared specimen, linked explicitly to a population or
locality, and one complete continuous ecology record per target. Every target
must have both views and at least one morphology member. Unequal counts are
retained. Independent targets are an analyst assumption; neither an ID match nor
locality membership verifies that assumption.

Select ordered analytical features explicitly. Use group-mean morphology PCA,
source-row ecology PCA, standard scaling, zero assignment threshold and identical
weights, view/joint dimensions, rank tolerance and preprocessing across arms.
A conflicting configuration is rejected instead of silently changed. Target means
are averaged before morphology standardization/PCA; native source membership
and projection remain intact. Ecology is never copied to individuals as independent
evidence. Missingness, count/categorical analysis and unmatched-view inference
are outside this interface.

## Versioned prespecified arms

The new schema-1 grouped configuration contains one to sixteen named arms, each
with `name`, `kind` (`ordinary` or `predictive`), a native integrative `model`, and
optional native `screen`, `bootstrap` and `fold_size_policy` objects. The included
example declares six arms. Existing model defaults, serializers, validators,
seed rules, candidate eligibility, fitting and significance procedures are unchanged.

Ordinary arms retain native supported bootstrap settings in their model. Ordinary
shared-covariance AIC stability remains unsupported. Predictive arms require an
explicit screen configuration with model bootstrap disabled. Full-family
predictive stability is unsupported. Shared predictive bootstrap additionally
requires locality targets and five fixed folds; population-target point screens
are supported but population predictive bootstrap is rejected. The existing
proportional-distinct fold-size policy is explicitly selectable only within its
native shared/locality scope; omission retains legacy behavior.

Prespecify configurations before viewing results. The unified report never chooses
a covariance family, promotes a default, creates a consensus species partition,
computes a new p-value or transfers historical significance to another selection.
HDBSCAN remains in the separate supported one-table interface.

## Reading the report and evidence

`report.html` is self-contained and works offline. It shows sampling counts,
selected/ordinary K, covariance family, selection rule, point/screen/stability
status, native screening decisions and conditional membership evidence. PCA
plots show the first two joint PCs; quantitative agreement uses the complete
partition, not the displayed axes or arbitrary label colors.

`manifest.json` binds both source-byte hashes, schema and identity mapping,
selected-value hashes, resolved settings and common cohort. `inputs/` retains
snapshots and fit-free ingestion; `groups.json` retains target counts
and specimen membership; `groups.csv` provides the count table. `models.csv`, `comparisons.json` and `comparisons.csv`
provide small normalized outputs. Each `arms/<name>/` directory preserves its
standalone workflow artifacts. Native BIC retains selected parameters and candidate
BIC diagnostics; native AIC additionally retains complete candidate parameter pools.
The adapter preserves these contracts instead of inventing missing evidence.

Screen failure does not become K=1. If a predictive screen is withheld but its
ordinary fit exists, the report keeps the ordinary result separately and excludes
the unavailable screened partition from pairwise comparisons. Other independent
arms continue. Failed draws are retained without replacements. Incomplete requested
analysis returns nonzero CLI status; point comparisons can remain available when
only stability is incomplete.

Agreement reports explicit jointly resolved unit/pair denominators and degeneracy.
A degenerate constant-partition ARI of one is not biological accuracy. The report
keeps membership, sampling stability and significance distinct. Whole-locality
predictive draws preserve members, counts and multiplicity, with original-ID folds
fixed so copies cannot leak across training and validation. All-unit evidence
includes resubstitution; OOB counts, coverage, bounds and minimum-support withholding
remain exactly those of the native workflow. Neither is a species probability.

## Bounded verification

The [results](grouped-analysis-results.md) describe 36 fresh synthetic datasets
across six laws at 64/256 localities, 2,568 main candidate searches including native
shared predictive stability and exact standalone replays, and a separate 364-search
preflight. The frozen configuration is `examples/grouped-analysis/validation.json`.
`validate.py` exposes explicit preflight/audit/gate/main phases, with a guarded
multiprocessing entry point. Detailed evidence belongs outside Git; publish only
small aggregates and figures. Three datasets/cell and 25 draws verify orchestration
and accounting, without establishing general power, false-positive control or
empirical species accuracy. Existing empirical sampling/reference gates remain closed
where their required evidence is unavailable. No author contact is performed.

The original 2 GiB storage gate was retained after its measured preflight
projection exceeded that cap. A user-approved budget-only amendment raised the
local cap to 4 GiB. Seeds, models, observations, draws, search counts, allowance
and free-space requirements were unchanged; no preflight fits were repeated.
