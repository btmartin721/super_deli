# Matched multimodal analysis including fixed-landmark shape

`analyze-matched-multimodal` compares prespecified native PCA/GMM analyses on
one complete, explicitly matched specimen cohort. Select fixed-landmark shape
and at least one of genomics, continuous morphology or individual ecology.
Genomic data are optional in this coordinator. Single-modality commands and the
older matched genomic/tabular commands remain supported with their old schemas.
No new estimator, automatic winner or p-value is introduced.

## Install and run the invented example

Use Python 3.11 or 3.12 from the checkout:

```bash
python -m pip install -e '.[clustering,shape,genomics]'
python examples/multimodal/smoke.py --output-dir /tmp/multimodal-example
```

Open `/tmp/multimodal-example/analysis/report.html` in a browser. The example
generates invented data locally; it downloads no empirical measurements.
Shape/tabular-only runs need `clustering,shape`; add `genomics` for SNPio inputs.

## Required inputs and assumptions

Use the existing schema-1 dataset declaration with explicit specimen IDs and
observation-to-specimen links. Each selected view must contain exactly one
observation for every declared specimen. Rows may appear in different orders.
Inputs are snapshotted and checksummed before fitting; there is no silent
intersection or removal of incomplete specimens.

- Shape: reviewed TPS (`tps`) or long-form coordinate CSV (`landmarks_csv`),
  declared landmark IDs/order and two or three coordinate dimensions. Coordinates
  must be complete and landmarks fixed and homologous. CSV rows use
  `observation_id,landmark_id,x,y` and optional `z` for 3D, with optional explicit
  `unit_id`. TPS uses the existing declared ID field. Landmark identity and
  homology are biological assumptions, not inferred from a file's appearance.
- Genomics, when selected: complete supported diploid biallelic calls through
  SNPio GenotypeData readers (VCF, PHYLIP, STRUCTURE, GENEPOP). Existing reader
  options and optional popmaps are retained; population labels are not features.
- Morphology/ecology: explicitly selected complete continuous feature columns.
  Ecological measurements must be individual-level; locality values cannot be
  copied onto fish/specimens and treated as independent measurements.

No image decoding/landmark detection, semilandmark sliding, allometry correction,
missing-data handling, repeated digitizations, grouped/unmatched integration or
species-reference evaluation is added here. Existing empirical evidence gates
remain unchanged. Upstream marker detection must supply reviewed coordinates.

## Prespecify native arms

Declare one singleton per selected view and an all-view integrative arm.
Optional subset integrative arms are allowed, up to eight total arms. Each
source kind may occur once. Singleton and integrated view PCA dimensions and
preprocessing declarations must agree. For shape, the effective GPA/reflection/
PCA policies must agree, including treating null and explicit `n_components`
equivalently when they specify the same fixed retained dimension.

For a dataset with `outline` and `traits` modalities and continuous `length`
and `depth` columns, this complete configuration uses existing native defaults
except the explicitly declared settings:

```python
from super_deli.io.config import write_json
from super_deli.models.integrative_config import IntegrativeConfig, ViewConfig
from super_deli.models.multimodal_analysis_config import (
    MultimodalAnalysisArm, MultimodalAnalysisConfig,
)
from super_deli.models.shape_cluster_config import ShapeClusterConfig
from super_deli.models.tabular_cluster_config import TabularClusterConfig

settings = dict(n_pcs=2, max_k=4, min_cluster_size=5, n_bootstrap=0,
                n_init=10, random_seed=301)
shape = ShapeClusterConfig(True, **settings)
traits = TabularClusterConfig(True, feature_names=("length", "depth"), **settings)
views = (ViewConfig("outline", "morphometric", shape=shape.shape),
         ViewConfig("traits", "morphological", feature_names=traits.feature_names))
combined = IntegrativeConfig(True, views=views, covariance_type="tied", **settings)
analysis = MultimodalAnalysisConfig((
    MultimodalAnalysisArm("shape", "shape", ("outline",), shape),
    MultimodalAnalysisArm("morphology", "tabular", ("traits",), traits),
    MultimodalAnalysisArm("combined", "integrative", ("outline", "traits"), combined),
))
write_json(analysis.to_dict(), "multimodal.json")
```

```bash
super-deli analyze-matched-multimodal dataset.json   --analysis-config multimodal.json --output-dir multimodal-run-01
```

API: `workflows.multimodal_analysis.run_multimodal_analysis(dataset,
analysis_config=analysis, base_dir=source_directory, output_dir=new_directory)`.
New directories are required. CLI exits 0 for complete evidence, 1 for a
partial/failed-arm report, and 2 for an invalid-input/source-integrity failure.
Failed arms remain unavailable and are never converted into K=1.

For genomic arms use the existing `BaselineConfig` and optional original-locus
`block_ids`. For ecological views set `ecology_measurement_scope="individual"`.
Prespecify view weights, joint PCs, K search and full/shared integrative covariance;
there is no search across these choices. Native genomic singleton covariance is
diagonal, shape/tabular singletons are full, and integrative covariance is
declared explicitly. Changes between arms are not integration effects alone.

## Reports and evidence

The offline `report.html` and ID-aligned `assignments.csv` show native candidate
clusters and maximum conditional memberships for each arm. Native complete
memberships and preprocessing live in `arms/<name>/`; source snapshots/cohort
accounting are in `inputs/` and `manifest.json`. `models.csv`,
`comparisons.csv/json` and `summary.json` retain status, search-limit flags,
label-invariant agreement, settings and failures. Labels/colors belong to each
arm, so compare partitions through the agreement table rather than label numbers.
All PCs enter fitting; the displayed two-dimensional plots are partial views.

GPA removes translation, proper rotation and positive scale under the declared
reflection policy. It does not remove size-related shape trends or digitization
error. Centroid sizes and source calibration metadata remain in native shape
artifacts; they are not automatically added as model features.

Native shape and integrative specimen bootstraps retain every landmark of each
resampled specimen and refit GPA, tangent PCA and GMM selection. Paired integration
resamples matching specimens across views, retaining multiplicities. Genomic
singleton stability instead resamples declared locus blocks with specimens fixed;
it requires blocks or an explicit unlinked assumption. Never pool these measures.
Native inclusion/OOB denominators, selected-K counts, failure bounds and failed
draws remain separate. Stability is not species probability or significance.

Use BIC only within one native search. Added or redundant views can dilute or
emphasize structure; fixed joint PCA can discard an informative view. No consensus,
model winner, confidence in species boundaries or new p-value is calculated.
See [the bounded validation and limitations](matched-multimodal-results.md).
