# Modular architecture

Super Deli is being redesigned for integrative species-delimitation research.
The foundation provides an installable package, data declarations, validation,
provenance, and result conventions. The second checkpoint adds SNPio genomic
ingestion. The third adds a genomic PCA/GMM baseline with explicit assumptions
and block-bootstrap stability; see [genomic baseline](genomic-baseline.md).
The fourth adds typed morphology/ecology tables; see [tabular ingestion](tabular-ingestion.md).
The fifth adds fixed-landmark TPS and coordinate CSV readers;
see [morphometric ingestion](morphometric-ingestion.md).
The sixth adds image manifests and PDF/PNG/JPEG/HEIC/HEIF display decoding;
see [image ingestion](image-ingestion.md).
The seventh adds explicit visible-marker localization and coordinate exports;
see [landmark detection](landmark-detection.md).
The eighth adds fixed-landmark GPA and tangent-space PCA with frozen-reference
projection; see [shape analysis](shape-pca.md).
The ninth adds full covariance GMMs on shape scores with whole-unit bootstrap;
see [shape clustering](shape-clustering.md).
The tenth adds numeric morphology/ecology PCA/GMM with preprocessing refits
and biological-unit bootstrap; see [tabular clustering](tabular-clustering.md).
The eleventh adds common-unit weighted score fusion, joint PCA/GMM and paired
bootstrap; see [integrative clustering](integrative-clustering.md).

## Boundaries

| Module | Responsibility |
| --- | --- |
| `core.data` | Immutable specimen, observation, feature, and dataset declarations |
| `core.schema` | Structural parsing and configuration version checks |
| `core.validation` | Biological references, feature declarations, and file availability |
| `core.provenance` | Configuration snapshots, software versions, and optional input hashes |
| `core.results` | Assignment and uncertainty reporting contracts for future estimators |
| `core.tabular` | Typed table records retaining feature schemas, biological levels and missingness |
| `core.morphometrics` | Fixed-landmark records, correspondence, missing masks and calibration metadata |
| `core.images` | Linked image records and derivative/source references |
| `core.detection` | Candidate marker geometry, point status and unresolved coordinates |
| `io.config` | Strict JSON reading and non-overwriting artifact writing |
| `io.genomic` | Optional SNPio readers, staged inputs, sample linkage, and ingestion artifacts |
| `io.genomic_encoding` | Conservative diploid SNPio encoding and source-locus indexing |
| `io.staging` | Shared isolated input copying and fingerprint verification |
| `io.tabular` | Strict CSV/TSV parsing, source-ID linkage, typed artifacts and coverage |
| `io.delimited`, `io.numeric` | Shared CSV/TSV structure and finite-decimal parsing |
| `io.tps`, `io.landmark_csv` | Fixed-landmark syntax and named coordinate rows |
| `io.morphometric_base`, `io.morphometric` | Shared coordinate/link rules and ingestion artifacts |
| `io.image_manifest`, `io.image_decode`, `io.images` | Image identity/selectors, optional codecs, staged sources and PNG artifacts |
| `io.detection_artifacts` | Separate visual overlays and ingestible coordinate exports |
| `io.landmark_export`, `io.shape_model` | Shared coordinate exports and validated numerical shape-model artifacts |
| `models.config` | Strict, dependency-free baseline settings and assumptions |
| `models.pca_gmm` | Numerical preprocessing, PCA/GMM search, and block stability |
| `models.mixture` | Shared fixed-space BIC/GMM selection and numerical-fit lock |
| `models.cluster_config`, `models.unit_bootstrap` | Shared independent-unit settings and complete-pipeline stability accounting |
| `models.tabular_cluster_config`, `models.tabular_gmm` | Numeric feature policies, preprocessing/PCA/GMM and training-only refits |
| `models.integrative_config`, `models.integrative` | Explicit target/view policies, raw PCA callbacks, profile fusion and paired target-unit bootstrap |
| `models.shape_cluster_config`, `models.shape_gmm` | Shape-clustering assumptions and complete-pipeline biological-unit bootstrap |
| `models.marker_config`, `models.marker_detection` | Dependency-free named rules and optional component-based localization |
| `models.shape_config`, `models.procrustes`, `models.shape_pca` | Explicit shape policies, GPA geometry and tangent-space ordination |
| `workflows.genomic_baseline` | Connect adapters/models to reproducible result artifacts |
| `workflows.landmark_detection` | Image ingestion, provisional marker coordinates, diagnostics and provenance |
| `workflows.shape_pca` | Fixed-landmark ingestion, fit/projection, exclusions and auditable shape artifacts |
| `workflows.shape_gmm` | Reuse shape ingestion/point fit and export candidate clusters and stability evidence |
| `workflows.cluster_artifacts` | Shared candidate evidence, linked assignments, mixture parameters and stability exports |
| `workflows.tabular_gmm` | One-table ingestion, biological levels and auditable preprocessing/clustering artifacts |
| `io.integrative_artifacts`, `workflows.integrative` | Source-to-target links, selected adapters, reconstruction parameters and integration evidence |
| `cli` | Argument handling and presentation of API results |

The core does not import scientific libraries or perform modality I/O. Readers
belong in `io`, numerical estimators in `models`, and artifact orchestration in
`workflows`. Root imports and dataset validation do not load optional numerical
libraries. The `baseline` extra supplies numerical dependencies; the `genomics`
extra supplies SNPio; `images` supplies Pillow, pillow-heif and PDFium. The
`landmarks` extra supplies NumPy/SciPy for visible-marker detection and is used
together with `images`. The `shape` extra supplies NumPy for GPA/PCA without
image, SNPio or scikit-learn dependencies. The `clustering` extra adds shape
and tabular mixtures without SNPio or image codecs. Empty plugin registries remain unnecessary.

The distribution discovers only packages under `src/`. Obsolete imputation
code, generated outputs and personal datasets are archived outside the
repository; synthetic examples and immutable published benchmark evidence remain.

## Configuration version 1

Configurations are UTF-8 JSON. Duplicate keys, unknown fields, nonfinite JSON
constants, unsupported schema versions, and malformed field types are errors.
Identifiers are nonempty, case-sensitive strings without surrounding whitespace.
They are never automatically trimmed, uppercased, or matched by row position.

Top-level fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Required integer `1` |
| `dataset_id` | Required dataset identifier |
| `random_seed` | Nonnegative integer; defaults to `0` |
| `specimens` | Specimen objects with `specimen_id` and optional `population_id`/`locality_id` |
| `populations`, `localities` | Explicit arrays of identifiers used by biological links |
| `modalities` | Required, nonempty array of source declarations |

At least one biological unit and one modality must be declared. Specimens are
optional for a dataset containing only population- or locality-level data.
Population and locality identifiers are sampling metadata, never species labels.
Known taxonomic labels and benchmark truth should remain separate evaluation
inputs rather than features used to discover the same labels.

Each modality has `name`, `kind`, `level`, `format`, `path`, and `observations`.
An observation has an `observation_id` identifying its source record and a
`unit_id` linking it to a declared biological unit at that modality's `level`.
Names are unique across modalities; observation identifiers are unique within a
modality. Multiple observations can refer to the same biological unit.

| `kind` | Declared `format` values | Additional declarations |
| --- | --- | --- |
| `genomic` | `vcf`, `phylip`, `structure`, `genepop` | Specimen level; optional `popmapfile`, default `null` |
| `morphological` | `csv`, `tsv` | Nonempty analytical feature schema |
| `ecological` | `csv`, `tsv` | Nonempty analytical feature schema |
| `morphometric` | `tps`, `landmarks_csv`, `image_manifest` | Coordinate or manifest reference |

The `validate` command checks declarations and references only. The separate
`ingest-genomic` command loads genomic contents through SNPio into `GenotypeData`,
with `popmapfile=None` when absent. `ingest-tabular` checks and loads morphology
and ecological CSV/TSV contents. `ingest-morphometric` reads fixed landmarks from
TPS/coordinate CSV. `ingest-images` checks image-manifest rows and decodes
selected pages/frames through the optional image libraries.
The optional `genomics` extra enables this dependency boundary on Python 3.11/3.12.
Genomic modalities may declare `reader_options` for STRUCTURE layout and
STRUCTURE/GENEPOP allele encodings; see [genomic ingestion](genomic-ingestion.md).

Tabular features declare `name`, `dtype`, optional `unit`, optional `categories`,
and `role` (`feature` or `covariate`, default `feature`). Types are `continuous`,
`count`, `binary`, `ordinal`, and `categorical`. Binary features require exactly
two states; ordinal categories require at least two states in their intended
order; unordered categorical features require at least one declared state.
Numeric features cannot declare categories. Units are recorded but not converted.
Covariates are recorded but no adjustment is performed.
The optional modality field `tabular_options` configures observation/unit ID
columns, exact missing tokens and explicitly ignored columns for morphology and
ecology. Column roles must be disjoint, and missing tokens cannot overlap category
states. Omitting this field preserves existing version-one configuration hashes.
The separate optional `morphometric_options` declares 2D/3D dimensions, ordered
landmark IDs, coordinate/scale units, TPS identity and missingness. These options
are required when ingesting coordinates. They do not change existing hashes when
omitted, and apply only to `tps`/`landmarks_csv` sources.
Optional `image_options` declares image IDs, nominal PDF DPI, pixel/source-byte
limits and HEIF bit-depth policy, only for `image_manifest`. Its omission likewise
preserves existing configuration hashes.

## Linking partial and unpaired data

The example in `examples/foundation/` contains measurements of two out of three
specimens, a repeated measurement of one specimen, and ecological observations
at two localities. The report preserves all declared specimens and separately
counts observations, unique biological units, repeat observations, and missing
units within each modality's level.

Different individuals can carry different modalities and share explicit population
or locality links. These links do not establish specimen correspondence. Group
measurements are not broadcast to individuals or counted as independent specimen
replicates. Joint inference from unpaired data needs a model whose assumptions
make it identifiable; the foundation performs no automatic fusion or aggregation.

Models will need explicit capabilities for modalities, inference levels, and
missingness. Accepting a configuration does not establish that any particular
future model can analyze it.

## Images and existing TPS utility

`image_tps_manifest.py` retains its existing metadata checks and tests, with
PDF/HEIC/HEIF extensions added to discovery. The source distribution retains the
script; the installed package uses separate ingestion modules. Its CSV columns remain:

`image_path`, `specimen_id`, `group`, `split`, `tps_path`, `landmark_count`,
`scale_value`, `scale_unit`, `check_status`, `check_details`.

The validator checks only manifest file references. The separate image adapter
checks row identity, explicit biological links and selected PDF pages/raster
frames, preserves original bytes and writes display-oriented PNG derivatives.
The existing schema is accepted with explicit `observation_id_column="specimen_id"`;
separate observation IDs support repeated views. Group/split/TPS/scale fields stay
metadata. The separate marker workflow localizes explicitly named visible marks
and exports provisional coordinates; anatomical learning and coordinate-frame
transforms remain later work. An image of a marker is not a TPS coordinate record.

## Provenance and output safety

Relative source paths resolve against the configuration file's parent directory;
absolute paths are accepted. Validation reads the configuration, checks declared
files are regular and readable, and writes nothing by default. With `--report`,
only a new report file is created; an existing path is never replaced. No output
directory is created automatically.

Reports embed the parsed configuration, its canonical-JSON SHA-256, random seed,
resolved base directory, timestamp, platform, Python version, and package version.
`--hash-inputs` hashes directly referenced data/popmap files in bounded memory.
Hashing is opt-in because genomic inputs can be large. A change detected during
hashing is an error. Image manifests are not recursively traversed. Hash status
explicitly distinguishes `computed`, `not_requested`, and `skipped_invalid_dataset`.
Hashes establish byte identity; they do not validate biological contents.

CLI exit codes are `0` for valid declarations/references, `1` for semantic or
reference errors, and `2` for malformed configurations, I/O errors, or usage errors.
Every completed validation report includes `contents_not_checked` as a warning.
Programmatic callers may use `check_files=False` for declaration-only checks;
the report records whether file checking was requested.

## Statistical reporting contract

`DelimitationResult` represents candidate assignments at one declared
biological level. It requires unique unit assignments and explicit evidence.
The configuration SHA-256 links a result to its configuration/provenance record.
An unresolved assignment uses `label=None` and a reason. Each evidence record
declares its target, kind, method, and status. Unavailable or uncomputed evidence
requires an explanation and cannot contain a numeric estimate.

Bootstrap frequencies require a replicate count. Membership probabilities remain
distinct from bootstrap frequencies. Confidence intervals require finite, ordered
endpoints and a confidence level. P-values require a null hypothesis and an
explicit multiplicity treatment, including `none` if unadjusted. These checks
enforce reporting structure; they do not establish statistical validity.

Each estimator checkpoint must independently validate its assumptions, uncertainty
calibration, and interpretation. Evaluation must account for genomic linkage,
spatial dependence, repeated observations, model/cluster selection, and training
leakage. Candidate clusters must remain distinguishable from validated species
hypotheses. Simulated truth is conditional on the generating model; empirical
reference labels need independent evidence and documented uncertainty.

## Checkpoint sequence

1. **Completed foundation:** Package, contracts, validation, provenance, tests, CI, and a draft PR.
2. **Completed:** SNPio ingestion and supported-format fixtures with optional popmaps.
3. **Completed:** Genomic PCA/GMM baseline, block stability, and bounded synthetic checks.
4. **Completed:** Typed tabular morphology/ecology ingestion with value/link fixtures.
5. **Completed:** TPS/coordinate CSV ingestion with coordinate/identity truth fixtures.
6. **Completed:** Image decoding with source preservation and synthetic display truth.
7. **Completed:** Visible-marker localization with explicit identity rules and unresolved points.
8. **Completed:** Fixed-landmark GPA, tangent PCA and frozen-reference projection.
9. **Completed:** Shape PCA/GMM candidate groups and whole-unit bootstrap stability.
10. **Completed:** Numeric morphology/ecology PCA/GMM with preprocessing and unit stability.
11. **Completed:** Common-unit weighted score integration, joint PCA/GMM and paired unit stability.
12. **Completed:** Truth-separated simulation benchmarks with dataset-level Monte Carlo uncertainty.
13. **Completed:** Larger prespecified benchmark, raw-artifact audit and results analysis.
14. **Completed:** Approximate single-Gaussian calibration in frozen integrative scores.
15. **Completed:** Full-pipeline Monte Carlo calibration under a declared raw-data reference law.
16. **Completed:** Matched/grouped empirical morphology/ecology nulls and boundary-aware covariance fitting.
17. **Completed:** Independent-SNP, linked-block and joint genomic–phenotypic fitted nulls.
18. **Completed:** Full/shared integrative and grouped clustering, group-mean PCA and scoped calibration.
19. **Completed:** Grouped AIC, full-pipeline likelihood-gain tests and larger false-positive/power studies.
20. **Completed:** Predictive-screened full-family AIC and N=64 recovery diagnostics.
21. **Completed:** Optional shared-covariance grouped AIC and predictive screening with paired validation.
22. **Completed:** Whole-locality bootstrap stability for fixed-fold shared predictive AIC.
23. **Completed:** N=64 shared predictive-bootstrap diagnosis with paired support/optimizer controls.
24. Empirical species-delimitation validation with independently supported biological reference groups.
25. Broader missing/unpaired/mixed-feature inference and calibrated uncertainty.
26. Anatomical landmark learning, semilandmark sliding and further model families.
27. Consolidated user guidance, end-to-end examples and a scoped research release.

The later calibration and diagnostic workflows remain restricted to their declared
assumptions and data types; completing their engineering does not establish species
boundaries. See the current README and linked result reports for precise validation
coverage. The shared predictive extension is described in
[shared-covariance predictive screening](shared-predictive.md), with a separate
[whole-locality stability workflow](predictive-stability.md).

The genomic baseline reports genetic clustering and locus-block sensitivity.
It does not supply taxonomic validation, formal species-boundary significance,
image decoders, empirical-data downloads, or expensive benchmark campaigns.
Tabular ingestion adds no estimators, preprocessing, fusion or significance tests.
The separate `cluster-tabular` baseline fits continuous/count analytical features
from one table and preserves its specimen/population/locality level. It uses
explicit transformations, missingness and scaling policies, shared BIC/GMM
selection, and independent-unit pipeline bootstrap. It adds no category encoding,
allometry correction, spatial resampling, modality fusion or species null test.
The separate integrative baseline reuses numerical view preprocessing without
fitting a view GMM. It aligns exact unit identities, permits explicit upward
specimen-to-group means, normalizes retained score profiles and fits joint
PCA/GMM. Paired target-unit draws refit every numerical stage. Missing modalities
are rejected or handled by explicit complete-case exclusion; no ecological
down-broadcast, inferred matching or species test is introduced.
Coordinate ingestion preserves source geometry/calibration and adds no image
decoding, alignment, distances, shape estimators or species significance tests.
The `shape-pca` workflow produces descriptive coordinates/ordination and explicitly
uncomputed uncertainty. The separate `cluster-shape` workflow adds candidate
shape groups and resampling stability, conditional on declared independence,
landmark and mixture assumptions; it adds no species test or calibrated species probability.

The separate [Gaussian score calibration](gaussian-score-calibration.md) checkpoint
adds a global fitted-null diagnostic and independent score-level validation. It
preserves original results and does not calibrate preprocessing/PCA selection,
cluster-specific support, or biological species boundaries.

The [full-pipeline reference test](full-pipeline-calibration.md) declares a known
four-view continuous-population law in `models.pipeline_config`. The workflow
streams fresh raw simulations through the existing integrative point pipeline;
it does not maintain a second preprocessing or clustering implementation.
`models.monte_carlo` shares upper-tail accounting with the Gaussian score test.
`benchmarking.pipeline_study` separates observed data and null seed streams;
`benchmarking.pipeline_analysis` audits archived bytes and reconstructs numerical
results without refits. Stability remains separate and empirical fitted-null
or species-boundary calibration is not supplied by this reference-law checkpoint.

The [empirical raw-Gaussian checkpoint](empirical-null-calibration.md) estimates
joint raw-feature nuisance parameters in `models.empirical_null`, before PCA.
The empirical workflow and known-reference workflow share point-fitting and
archival helpers in `workflows.calibration_artifacts`; neither duplicates the
integrative estimator. The study/auditor reuse dataset accounting and selected
pipeline numerical checks. Fitted nulls and known reference laws retain separate
policies, evidence targets, assumptions and interpretation.


The [small-sample bootstrap diagnostic](bootstrap-instability.md) is a separate
benchmarking layer. It reuses unchanged production point/occurrence workflows,
retains original-ID size accounting and runs truth-informed optimization only
after ordinary fits. New diagnostic configuration, gate accounting, post-fit
metrics, immutable studies and independent audits are separate modules; none
change production defaults or historical p-values. Detailed evidence stays local.
