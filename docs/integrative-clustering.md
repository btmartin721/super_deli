# Integrative clustering of linked biological units

The separate [simulation benchmark](integrative-benchmarking.md)
around this unchanged estimator. It evaluates generating-partition recovery and
stability error without treating simulation labels as empirical species truth.

```bash
python -m pip install -e '.[dev,clustering,genomics]'
super-deli cluster-integrative examples/integrative/grouped_dataset.json \
  --model-config examples/integrative/grouped_model.json \
  --output-dir /tmp/super-deli-integrative-grouped
super-deli cluster-integrative examples/integrative/matched_dataset.json \
  --model-config examples/integrative/matched_model.json \
  --output-dir /tmp/super-deli-integrative-matched
```

Python 3.11/3.12 supports two to four selected views, at most one of each kind.
Use the existing unimodal commands when only one view is available. `clustering`
alone supports non-genomic integration; add `genomics` for SNPio input. Image
codecs are not required: shape views consume reviewed TPS/landmark coordinates.

## Biological alignment

The required JSON `target_level` is `specimen`, `population` or `locality`.
Candidate targets are the union observed in selected views. Declared units with
no selected observations are reported separately as unobserved targets.
Same-level views match exact, declared unit IDs; source row order and observation
names do not imply correspondence. Every target must occur in every selected
view. Default `missing_views="error"` rejects unequal target coverage. Explicit
`complete_cases` restricts numerical fitting to the intersection and records
excluded units and their missing views. It does not impute a missing modality.
Selection can produce a biased complete-case cohort; exclusion reporting does
not correct that bias.

Different individuals can contribute to a common population or locality only
when each specimen has the corresponding explicit `population_id`/`locality_id`
in the dataset and that view declares `aggregation="mean"`. The fitted profile
is the arithmetic mean of the source observations' **PC scores**, after view
preprocessing/PCA. Raw landmark coordinates are not averaged before GPA, and
genotypes are not converted into fractional individual calls. Each original
source observation has equal weight in its view's raw PCA fit and in its
target's mean profile. Therefore groups with more source observations have more
influence on that view's fitted basis. Their integrated profiles subsequently
have equal weight in joint PCA and GMM fitting.
Aggregation can blend heterogeneous biological groups and removes within-group
variation. Resulting population/locality clusters cannot be transferred back
to individual species assignments.

Direct population/locality views must already be at the target level, with
`aggregation="none"`. No locality-to-specimen broadcast, inferred pairing,
population-to-locality conversion, category matching or taxonomic-label matching
is performed. Technical repeats with duplicate source unit IDs are rejected;
aggregate them under an external measurement protocol first. SNPio popmaps stay
optional. Integration group links come from the dataset, not inferred popmap labels.

## View pipelines and fusion

| View kind | Pipeline |
| --- | --- |
| `genomic` | SNPio VCF/PHYLIP/STRUCTURE/GENEPOP ingestion and diploid encoding, locus filtering, explicit missingness/scaling, PCA |
| `morphological`, `ecological` | Typed numeric analytical subset, optional count transformation, filtering, explicit imputation/scaling, PCA |
| `morphometric` | Complete fixed landmarks, GPA, tangent projection and shape PCA |

These reuse the existing [genomic](genomic-baseline.md), [tabular](tabular-clustering.md)
and [shape](shape-pca.md) preprocessing rules. Genomic views require declared
`ploidy=2`. Tabular features are continuous/count measurements; covariates and
category labels are excluded or rejected by the existing feature contract.
Shape homology and coordinate-axis conventions remain input assumptions.
Optional shape `drop_observations` removes incomplete observations before
coverage alignment and records their identities. Other cell-level missingness
requires each view's explicit policy; no entirely unobserved row is filled.

Each view retains a fixed, prespecified `n_pcs`. Insufficient numerical rank
fails instead of reducing dimension or discarding the view. Per-view GMMs are
not used for fusion: the retained score profiles preserve continuous information.

For N target profiles in view v, let `S_v` be its N-by-r score matrix. Define
`mu_v = mean(S_v)`, `a_v² = sum((S_v-mu_v)²)/(N-1)` and normalized positive
prespecified weights `pi_v = w_v/sum(w)`. The fused matrix is the column
concatenation of `sqrt(pi_v)*(S_v-mu_v)/a_v`. Thus each block's total sample
variance is `pi_v`; a view's raw scale or number of source features cannot by
itself set the total block energy. All retained axes within a view keep their
relative variances. Weights outside representable relative float64 range fail
instead of silently zeroing a view.

Full-SVD joint PCA reduces this fused matrix to global `n_pcs`, without whitening.
BIC selects among size-admissible full covariance GMMs K=1..max_k in that fixed
joint space. Minimum component sizes count distinct target units. Any feasible
candidate's numerical/convergence failure aborts the search. BIC is not compared
across dimensions, configurations or replicate spaces.

This is an explicitly specified weighted-score baseline. Multiblock PCA has
many alternative weight and component definitions, discussed in
[Westerhuis et al. (1998)](https://doi.org/10.1002/(SICI)1099-128X(199809/10)12:5%3C301::AID-CEM515%3E3.0.CO;2-S).
This implementation does not claim to implement MFA, JIVE, CCA or a calibrated
species likelihood. Preset weights describe feature-space geometry; they are
not probabilities or statistical evidence weights. Correlated or overlapping
views can emphasize a shared signal. PCA truncation, missingness, outliers,
allometry and non-Gaussian density can change candidate components.

## Configuration

```json
{
  "schema_version": 1,
  "assume_independent_units": true,
  "target_level": "locality",
  "missing_views": "error",
  "n_pcs": 2,
  "max_k": 4,
  "min_cluster_size": 5,
  "n_bootstrap": 100,
  "random_seed": 71,
  "views": [
    {"modality": "markers", "kind": "genomic", "ploidy": 2,
     "n_pcs": 2, "weight": 1, "aggregation": "mean"},
    {"modality": "traits", "kind": "morphological",
     "n_pcs": 2, "weight": 1, "aggregation": "mean"},
    {"modality": "environment", "kind": "ecological",
     "n_pcs": 2, "weight": 1, "aggregation": "none"},
    {"modality": "outline", "kind": "morphometric",
     "n_pcs": 2, "weight": 1, "aggregation": "mean",
     "shape": {"schema_version": 1}}
  ]
}
```

Global fields also include the shared GMM controls (`n_init=10`, `max_iter=500`,
`tol=0.001`, `reg_covar=1e-6`, `assignment_threshold=0`, `n_threads=1`), numerical
`rank_tol=1e-12`, `max_working_bytes=268435456` and `max_pairwise_bytes=268435456`.
`min_cluster_size` must exceed global `n_pcs`.

Each view requires `modality` and `kind` matching the dataset. Defaults are
`n_pcs=2`, `weight=1`, `aggregation="none"`, `missing_policy="error"` and
`scaling=null`. Null scaling means Patterson for genomic data and standard
scaling for tabular data. Genomic scaling also accepts `center`/`standard`;
tabular scaling accepts `center`/`standard`. Tabular-only options are
`feature_names=[]` (all analytical columns) and `count_transform="identity"`
or `log1p`. Morphometric views require the explicit existing `shape` object;
its component count must be null or agree with view `n_pcs`, and its repeated
policy must reject. Inapplicable genomic/tabular/shape options are rejected.
Source eligibility and numeric validation occur before fitting; unknown fields,
invalid types and absent biological links fail.

## Paired bootstrap and statistical scope

Each bootstrap draws N independent **target units** with replacement, using
the same draw across every view. For aggregated views, each target occurrence
contributes all its observed source members with their multiplicities. Refits
learn view filtering/imputation/scaling/PCA, GPA where applicable, group profiles,
block normalization, joint PCA, all GMM candidates and BIC selection. The fitted
replicate then projects all original retained targets through its frozen
pipeline. Arbitrary cluster labels are handled through coassignment/exact-set
recovery, not component matching across different bases.

The [shared unit bootstrap](shape-clustering.md) retains every attempted draw
in requested denominators and records seeds, paired target draws, source-row
draws, diagnostics and partitions or failures. In-bag predictions can be
optimistic. OOB pair frequencies include only draws excluding both target units
and have explicit requested/successful denominators. Unobserved or
failure-affected OOB pairs have no point frequency. Failed-outcome bounds
`[s/B,(s+f)/B]` identify unknown failed assignments; they are not confidence
intervals. Any failed draw withholds the main and exact-recovery frequencies.
`n_bootstrap=0` explicitly reports uncomputed stability.

Independence concerns the chosen target units. Resampling holds their observed
within-group members fixed and does **not** estimate within-population sampling
error. Genomic source-locus eligibility/encoding is fixed during ingestion;
genomic means, scaling and PCA are refitted, but locus blocks are not resampled
here. This is target-unit pipeline sensitivity conditional on the observed
members, loci and feature definitions. It is distinct from the genomic command's
locus-block sensitivity. Spatially dependent sites or related populations can
violate the target-independence assumption. No spatial/hierarchical sampling
model, environmental replication correction, measurement-error model or
allometry adjustment is supplied.

Integrated components remain candidate structure. Conditional membership,
bootstrap frequency and species significance are separate evidence types.
There is no species-boundary null test, calibrated species probability or
confidence interval in this checkpoint. Finite bootstrap counts also have
Monte Carlo error; the ten-draw examples are bounded software checks.

## Artifacts, reconstruction and API

| Artifact | Contents |
| --- | --- |
| `view_NNN/ingestion/` | Isolated existing ingestion outputs and source hashes |
| `view_NNN/source_values.npy`, `source_rows.json` | Prepared numeric values, exact source/target links, feature/landmark order and numeric-array hash |
| `view_NNN/encoding.json` | Genomic SNPio encoding and source-locus exclusions, when applicable |
| `view_NNN/parameters/` | View PCA parameters or validated shape model, target score profiles, block mean/scale/weight and source membership |
| `input_coverage.json` | Coverage and excluded incomplete observations, including when fitting later fails |
| `alignment.json` | Retained target order, complete-case exclusions and exact per-view source members |
| `fusion.npz`, `scores.csv` | Fused target score blocks, joint PCA parameters and linked joint scores |
| `mixture.npz`, `assignments.csv`, `result.json` | Canonical mixture parameters, assignments, conditional membership and scoped evidence |
| `candidates.json`, `bootstrap_replicates.json` | K search, draw seeds/indices, refit diagnostics, partitions or failures |
| `stability.npz`, `clusters.json` | Coassignment/OOB counts and denominators, exact recovery and failed-outcome bounds |
| `summary.json` | Completion, levels, selected K, policies, interpretation, versions and hashes |

NPZ/NPY files use numeric arrays and `allow_pickle=False`. View summaries map
`retained_source_indices` and point `training_source_indices` to that view's
`source_values.npy`, which precedes complete-case filtering. Bootstrap
`source_training_indices` address the retained per-view row order and are mapped
through `retained_source_indices`. Shape-missing exclusions precede those
prepared arrays and remain identifiable by observation ID. This prevents row
indices from becoming ambiguous after source-order changes or exclusions.

Top-level assignment `observation_id` denotes the derived target profile and
equals its unit ID; original observations are in the explicit membership maps.
Pairwise axes follow `summary.unit_ids`. Main coassignment is entirely NaN after
any failed draw; OOB NaNs mean unobserved or failure-affected pairs. Existing
directories cannot be overwritten. Point failures leave `failure.json` without
a completion summary; successful point fits can have unavailable stability.
PCA budgets apply per numerical fit, including expanded grouped bootstrap draws;
dense pairwise work uses a conservative `128*N*N` guard. Stored views, raw arrays
and O(B*N + B*source_rows + B*Kmax) records require additional memory/storage.

```python
from super_deli import load_config
from super_deli.io.config import read_json
from super_deli.models.integrative_config import IntegrativeConfig
from super_deli.workflows.integrative import run_integrative

summary = run_integrative(
    load_config("examples/integrative/grouped_dataset.json"),
    model_config=IntegrativeConfig.from_dict(
        read_json("examples/integrative/grouped_model.json")
    ),
    base_dir="examples/integrative",
    output_dir="/tmp/super-deli-integrative-api",
)
```

The numerical API `models.integrative.fit_integrative(inputs, config=...)`
accepts `IntegrationInput` objects containing prepared selected numeric arrays,
unique source identities and explicit target IDs. Use the workflow for validated
file ingestion, feature roles and linkage provenance. No saved-run prediction
CLI is provided; saved arrays permit independent reconstruction without Python
object serialization.

The fixtures have four generating groups defined by two complementary binary
signals: genomic/ecological views carry one signal and trait/shape views the
other. The grouped scenario uses 64 localities and different specimens for each
individual-level view; the matched scenario explicitly retains 64 of 128
individuals with all views. Ground truth is stored separately and never staged
as a feature. Tests cover four-group recovery, null structure, block normalization,
ID/order alignment, paired source draws, excluded-row isolation, parameter
reconstruction and real SNPio ingestion without a popmap. These checks are not
empirical species truth or biological uncertainty calibration.
