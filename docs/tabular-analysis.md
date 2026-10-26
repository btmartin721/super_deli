# Unified tabular analysis and model comparison

`analyze-tabular` runs prespecified models on one frozen, complete continuous
morphology or ecology table. It retains each model's native evidence and produces
one offline HTML report, machine-readable summaries and comparison tables.

Run from the repository root:

```bash
python -m pip install -e '.[clustering]'
super-deli analyze-tabular examples/tabular-analysis/dataset.json \
  --analysis-config examples/tabular-analysis/analysis.json \
  --modality traits --output-dir /tmp/super-deli-tabular-comparison
```

Use a new output directory on every invocation. Dataset paths resolve relative
to the dataset JSON; output paths resolve relative to the invoking directory.
The example contains invented identifiers and synthetic traits. Full study
inputs, fitted arrays and resampling records are kept outside the repository.

```python
from pathlib import Path
from super_deli import load_config
from super_deli.io.config import read_json
from super_deli.models.tabular_analysis_config import TabularAnalysisConfig
from super_deli.workflows.tabular_analysis import run_tabular_analysis

source = Path("examples/tabular-analysis/dataset.json").resolve()
summary = run_tabular_analysis(
    load_config(source),
    analysis_config=TabularAnalysisConfig.from_dict(
        read_json(source.parent / "analysis.json")
    ),
    base_dir=source.parent,
    modality_name="traits",
    output_dir="/tmp/super-deli-tabular-python",
)
```

## Configuration and boundaries

The separate schema-1 configuration contains one to sixteen named arms. Names
are safe lowercase identifiers. Each arm declares `family: gmm` or `hdbscan`
and its native model configuration. All arms must explicitly select the same
ordered features, PCA dimension and rank tolerance. The workflow requires
standard scaling, complete continuous selected analytical features, unique
biological units and a declared independence assumption. It accepts specimen,
population or locality rows, without creating groups or copying environmental
values onto specimens. Other nonselected columns retain their ingestion roles.

GMM arms use the existing full-covariance, BIC-selected K search. HDBSCAN arms
can vary explicitly declared density settings. Conflicting preprocessing is
rejected; no settings are silently changed. Existing standalone commands,
defaults, configurations, random-seed behavior and p-values retain their behavior.
This interface does not include grouped predictive AIC, integrative clustering,
count/categorical features, missing-data treatment or empirical evidence gates.

## Reading the outputs

- `manifest.json` binds original/snapshot source hashes, original/frozen dataset
  declarations and resolved model settings. Every arm reads the same snapshot.
- `summary.json`, `models.csv`, `comparisons.json` and `comparisons.csv` contain
  model status, K, noise/unresolved coverage and label-invariant agreement.
- `report.html` is self-contained and works offline. PCA colors are local to each
  model; gray denotes noise or unresolved assignments. Numerical comparisons
  use cluster membership, not color or the arbitrary label values.
- `arms/<name>/` preserves native fitted parameters, candidates, assignments,
  preprocessing and resampling evidence. `comparison-point.json` records the
  aligned partition and resolved mask used for comparison.

Agreement is conditional on jointly resolved, non-noise units. Read its unit and
pair denominators and joint coverage. Fewer than two joint units gives undefined
agreement. A degenerate ARI of one for two constant partitions follows the usual
mathematical convention and does not show recovered structure. All-noise
HDBSCAN succeeds with K=0; noise and unresolved units are not a shared cluster.
GMM raw candidate partitions are retained separately from assignment-threshold
resolution. K counts fitted candidate clusters, including clusters whose units
may all be unresolved by a declared threshold.

GMM posterior membership and HDBSCAN density strength are different measures.
GMM bootstrap stability refits with replacement and projects all original units;
its OOB evidence uses separate pair-specific denominators. HDBSCAN subsampling
refits without replacement, labels only included units and retains inclusion
counts and failure bounds. These supports are neither pooled nor ranked.
Parameter sensitivity is different from sampling uncertainty. Reported stability
is not a species probability or a significance test.

An arm failure leaves evidence, does not become K=1 and does not block independent
arms. Failed resamples are retained without replacement. Point comparisons remain
available when only stability is incomplete; a failed or mismatched point is
excluded from its comparisons. The overall status is partial/failed and CLI exit
status is nonzero for an incomplete requested analysis. Input/cohort failures
raise an error and leave `failure.json` once the output directory exists.

No model is declared the winner. No consensus species partition, new p-value,
comparison of BIC across model families, or general species-boundary claim is
provided. See [validation results](tabular-analysis-results.md).

## Frozen validation

`examples/tabular-analysis/validation.json` records all seeds/settings. The
explicit `validate.py` phases run preflight, audit, storage gate and main study.
Use an output root outside Git. The study has 36 observations across six laws
and N=64/256, four named arms, 672 main pipeline runs (including 480 family-native
resamples and 48 direct point replays), and a separate 88-run preflight. A GMM
pipeline includes its complete K search. Synthetic truth is never used to choose
settings. This suite verifies orchestration and descriptive behavior; three
datasets per cell cannot establish general scientific performance.

The executed checkpoint retained its original 512 MiB budget declaration and
failed storage gate. A user-approved budget-only amendment raised the local cap
to 1 GiB, with no additional fits or changes to seeds, settings or datasets.
