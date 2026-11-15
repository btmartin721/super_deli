# Matched genomic–phenotypic discovery comparison

The opt-in `analyze-matched-genomic` command compares existing PCA/GMM analyses
on one explicitly matched specimen cohort. It discovers candidate structure;
it does not name species, select a winning model or compute p-values.

## Install and try the invented example

Use Python 3.11 or 3.12 from the repository checkout:

```bash
python -m pip install -e '.[genomics,baseline,clustering]'
python examples/matched-genomic/smoke.py --output-dir /tmp/matched-genomic-example
```

Open `/tmp/matched-genomic-example/analysis/report.html` in a browser.
The example generates invented inputs; it does not download empirical data.

## Prepare real inputs

Use the existing schema-1 dataset JSON. Declare specimen IDs, one genomic
modality, and one morphological and/or ecological modality at `specimen` level.
Link each source observation/sample ID explicitly to its specimen. The genomic
source goes through SNPio's GenotypeData readers: VCF, PHYLIP, STRUCTURE or
GENEPOP. Reader-specific options remain in `reader_options`. `popmapfile` is
optional and defaults to None in SNPio. Population labels are not predictors.

Each selected modality must have exactly one observation for every declared
specimen; source row order may differ. This first coordinator requires complete
supported diploid biallelic genotypes and complete selected continuous traits.
Monomorphic loci still pass to native filtering. Missing/unsupported calls,
missing selected traits, repeated observations or incomplete cohorts stop before
fitting. Extra undeclared source records are rejected by native ingestion.
Ecology must have been measured on the individual, not copied from a locality.
There is no implicit intersection, group aggregation or missing-view imputation.

Declare all native model settings before looking at results. This complete
example writes a three-arm comparison for modalities named `snp` and `traits`,
with continuous morphology features named `length` and `depth`:

```python
from super_deli.io.config import write_json
from super_deli.models.config import BaselineConfig
from super_deli.models.genomic_analysis_config import (
    GenomicAnalysisArm, GenomicAnalysisConfig,
)
from super_deli.models.integrative_config import IntegrativeConfig, ViewConfig
from super_deli.models.tabular_cluster_config import TabularClusterConfig

settings = dict(n_pcs=2, max_k=4, min_cluster_size=5, n_bootstrap=0,
                n_init=10, random_seed=301)
genomic = BaselineConfig(2, "error", True, **settings)
traits = TabularClusterConfig(True, feature_names=("length", "depth"), **settings)
views = (ViewConfig("snp", "genomic", ploidy=2),
         ViewConfig("traits", "morphological", feature_names=traits.feature_names))
combined = IntegrativeConfig(True, views=views, covariance_type="tied", **settings)
analysis = GenomicAnalysisConfig((
    GenomicAnalysisArm("genomic", "genomic", ("snp",), genomic),
    GenomicAnalysisArm("morphology", "tabular", ("traits",), traits),
    GenomicAnalysisArm("combined", "integrative", ("snp", "traits"), combined),
))
write_json(analysis.to_dict(), "comparison.json")
```

```bash
super-deli analyze-matched-genomic dataset.json   --analysis-config comparison.json --output-dir matched-discovery-01
```

The API is `workflows.genomic_analysis.run_genomic_analysis(dataset,
analysis_config=analysis, base_dir=source_directory, output_dir=new_directory)`.
It returns the summary dictionary. Existing output directories are never
overwritten. CLI exit 0 means all requested native evidence completed; exit 1
means a partial/failed arm report, and exit 2 means invalid input or a fatal
source-integrity error. Failures are retained, never converted into K=1.

For both morphology and ecology, declare a singleton arm for each and an
all-view integrative arm. A phenotype-only integrative arm is optional. Set
`ecology_measurement_scope="individual"`. The config permits three to five
prespecified arms, keeping view PCA dimensions/scaling/selected features
consistent between singleton and combined declarations. Integrative weights,
joint PC count and full/shared covariance remain explicit native settings.
The genomic-only baseline has diagonal covariance and tabular singletons full
covariance, so differences cannot be attributed to integration alone.

## Read the output

- `report.html`: offline report with per-arm projections, agreement and native
  settings/evidence. Component colors and integer labels are local to each arm.
- `assignments.csv`: one ID-aligned specimen row, source observation IDs, native
  zero-based cluster labels and maximum conditional memberships. Missing failed
  arms stay blank. Membership is not a species probability or bootstrap support.
- `models.csv`, `comparisons.csv/json`, `summary.json`: status, selected K,
  search-limit flags and label-invariant agreement, without model ranking.
- `manifest.json`, `inputs/`: original provenance, verified snapshots (including
  optional popmaps), complete cohort and source-locus count.
- `arms/<arm>/`: unchanged native outputs, preprocessing, candidate criteria,
  complete memberships, replicate records, failures and native unit ordering.
  Genomic complete memberships are in `membership.npy`, its IDs in
  `summary.json`; tabular/integrative arrays are in `mixture.npz`.

Use BIC only within each native search, never across different fitted spaces.
Correlated views, weighting and PCA compression can hide or emphasize structure.
Agreement is agreement between fitted partitions, not evidence of biological
truth. Independence is an explicit assumption, not established by ID matching.

## Stability stays separate

Set each native model's `n_bootstrap` explicitly. Genomic stability requires
`block_ids` in original source-locus order, or an explicit `assume_unlinked=True`.
The length check occurs before locus filtering. The immutable map is recorded
in the analysis configuration. No block independence is inferred automatically.

The genomic baseline resamples whole locus blocks while holding specimens
fixed. Tabular models resample specimen rows. Integrative models resample paired
specimens across all views, holding source loci/eligibility fixed. Native refits
redo preprocessing and PCA. Their inclusion/OOB denominators and failure bounds
remain native; do not pool frequencies or treat these resampling schemes as
interchangeable. Disabled stability remains `not_computed`.

No neural model, significance workflow, missing-data, shape or grouped/unmatched
adapter is added. Existing defaults, historical p-values and empirical evidence
gates remain unchanged. See [bounded validation](matched-genomic-results.md).
