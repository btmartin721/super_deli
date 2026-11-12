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
24. **Completed:** Opt-in exposure-scaled fold sizes with paired whole-locality stability validation.
25. **Foundation completed:** Fit-free empirical data curation, source locks, anonymous matched manifests and a frozen non-executable protocol.
26. **Assessment completed:** Source-bound sampling/reference evidence review; ingestion and descriptive labels admitted, independence-dependent uses explicitly blocked and grouped screening inapplicable for the capture source.
27. **Audit completed:** Juvenile-skate replacement source, typed morphology/fixed-landmark representations and segregated COI reference diagnostics; voucher/haul correspondence remains insufficient for an independent benchmark.
28. **Assessment completed:** Bounded public-record skate correspondence review, source-bound mapping ledgers and local records-needed schemas; no authoritative specimen/voucher or specimen/haul bridge established, and no contact performed.
29. **Audit completed:** Groundwater-amphipod continuous traits, source-bound workbook/COI/locality ledgers and typed ingestion; independent sampling and reference truth remain unestablished, with no empirical fitting or contact.
30. **Assessment completed:** Source-bound amphipod sampling/reference review, three additional documentary voucher/COI corroborations and explicit conflicting/unresolved evidence; independent-unit fitting and species-boundary validation remain blocked.
31. **Completed:** Separate opt-in continuous tabular PCA/HDBSCAN, noise-aware subsampling and paired simulation validation.
32. **Completed:** Frozen noisy-moons density diagnosis, hierarchy/connectivity audits and archived replay parity; no production/default change.
33. **Completed:** Unified one-table PCA/GMM and PCA/HDBSCAN orchestration with coverage-explicit comparison and separate native stability.
34. **Completed:** Unified grouped morphology/ecology comparisons with explicit counts, native selection/stability boundaries and standalone parity.
35. **Completed:** Freeze the fit-free amphipod morphology pilot protocol and four trait/sex feasibility rosters. Independent raw-cell accounting verifies the sparse molecular overlap; executable fitting, resampling, significance and species validation remain blocked pending justified sampling/dependence and reference evidence.
36. **Baseline completed:** Opt-in one-table complete-case/observed-mean policy comparison with refitted native stability, frozen simulation sensitivity and explicit retention/geometric-loss limits. Broader missing/unpaired/mixed-feature models and calibrated missing-value uncertainty remain pending.
37. **Baseline completed:** Opt-in observed-coordinate GMM for incomplete continuous tables, with constrained conditional-moment EM, point-only BIC selection, retained failure evidence and raw-feature simulation comparisons. Resampling, significance calibration and grouped/integrative observed-data adapters remain pending.
38. **Diagnostic completed:** Frozen observed-coordinate GMM recovery/convergence diagnosis with original evidence preservation, bounded saved-path continuations, independent fresh confirmation and isolated known-parameter starts. No new default, covariance family or uncertainty claim.
39. **First neural baseline:** Opt-in complete continuous tabular autoencoder/HDBSCAN, numeric frozen transforms, explicit noise and refitted whole-unit subsampling. Prespecified initialization comparisons remain separate from sampling stability; no significance adapter or new default.
40. **Diagnostic completed:** Archived neural curved-moons geometry, neighborhood, hierarchy and selection diagnosis with fixed saved weights, exact density replays and independent audits. No new training, model/default promotion or significance claim; mapping/noise/metric effects remain partly entangled.
41. **Opt-in neural comparison completed:** Fixed-corruption denoising autoencoder/HDBSCAN with matched feature-noise controls, all three seeds, native subsampling, independent training/draw/replay audits and conservative pilot uncertainty. Curved-moons recovery remains weak; no default or significance change.
42. **Diagnostic completed:** Frozen supplied-K spectral graph/representation comparisons, without new training or unsupervised count claims.
43. **Experimental model and validation completed:** Raw-trait unknown-K eigengap workflow with explicit abstentions. Fresh validation found rank-two Gaussian oversplitting and no correct moons counts; the workflow is not recommended for automatic delimitation.
44. **Benchmark-only feasibility completed:** A frozen Gaussian-reference graph-count rule reduced the tested Gaussian oversplitting but still missed moons groups, often returning K=1. No public model adapter, default or historical p-value changed; no-supported-split is not evidence of one population. See [results](gaussian-reference-spectral-results.md).
45. **Unresolved:** A defensible group-count criterion before spectral resampling or user-facing reference-rule promotion; any follow-up requires a separately approved checkpoint.
46. **Opt-in matched neural baseline:** Complete continuous individual morphology/ecology with strict ID alignment, view-balanced reconstruction, frozen numeric transforms and matched native subsampling. The [paired pilot](matched-neural-results.md) retains seed differences, uninformative-view losses and weak moons recovery; no species/significance claim or default change.
47. **Completed:** Separate matched-view encoders, prespecified weights and frozen-coordinate diagnosis, with legacy parity and independent numerical audits.
48. **Opt-in model and fresh validation completed:** Separate encoders with a shared-covariance BIC GMM, frozen transforms, conditional memberships and matched subsampling. Standard four-group recovery improved, while Gaussian splitting and small-sample unequal-spread/curved-group limitations remain explicit. PCA controls preclude claiming a neural advantage; no default or significance change.
49. **Completed:** Full-pipeline separate-encoder/shared-GMM Gaussian calibration and larger false-split validation, with approximate departure-test interpretation and preserved defaults.
50. **Completed:** Frozen paired neural/PCA power and group-recovery validation, exact archived replays, approved runtime restoration and separate detection/recovery uncertainty. Weak signals remain unresolved; no neural advantage or default promotion is established. See [results](neural-power-results.md).
51. **Diagnostic completed:** Source-bound weak-signal overlap, affine retention, GMM eligibility/BIC and isolated truth-start diagnosis. Known-law reference agreement is not a formal ARI ceiling; PCA retains nearly the same known-law accuracy and no neural advantage is established. See [results](weak-signal-results.md).
52. **Opt-in supervised reference model completed:** Complete matched continuous morphology/ecology with a fixed residual classifier, separate ID-bound labels and frozen numeric prediction. The nonlinear pilot separates known-class assignment from discovery, retains nuisance-shift/unfamiliar-class limits and establishes no consistent residual advantage. See [results](residual-nonlinear-results.md).
53. **Opt-in reference-abstention layer completed:** Separately calibrated raw-distance/probability compatibility sets preserve original residual predictions and distinguish accepted, outside-support, ambiguous and classifier-conflict outcomes. The fresh pilot missed every joint operating target; coverage costs, nearby/overlapping groups and nuisance shifts preclude general novelty or safer-assignment claims. No default or p-value changed. See [results](reference-abstention-results.md).
54. **Opt-in class-local reference support completed:** Fifth-nearest original training neighbors, independent classwise calibration and frozen residual prediction preserve existing decision meanings. The 80-fresh/80-archived pilot met no primary joint or benefit target; ambiguity, unfamiliar acceptance and shifted conditions remain limiting. No policy/default or p-value change. See [results](local-reference-results.md).
55. Grouped/unmatched, missing-data, genomic and shape neural adapters, anatomical landmark learning, semilandmark sliding and further model families.
56. Consolidated user guidance, end-to-end examples and a scoped research release.

The [tabular neural workflow](tabular-neural.md) reuses typed ingestion and a
shared density-assignment helper. Raw-feature and latent standardization are
fitted separately within every whole-unit subsample. Its dependency-free
configuration and lazy CLI dispatch keep PyTorch optional. Saved numeric weights
support independently auditable encoder/decoder transforms without Python-object
serialization; new measurements receive no invented HDBSCAN species prediction.
Existing [density clustering](tabular-density-clustering.md), GMM workflows,
configuration hashes and p-values keep their behavior. The frozen simulation
comparison and independent audits are [documented separately](tabular-neural-study.md).

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


The [scaled fold-size policy](scaled-fold-stability.md) is an explicit typed
configuration, separate from the integrative model and screening schemas. Optional
hooks reuse the production fit/search and preserve the absent-policy path. A
fold-only mixture settings object separates effective training support from the
unchanged full-model minimum and dimension guards. Identity-derived thresholds
and mixture settings are saved before estimation. The generic bounded-study
runner, paired benchmark and independent archive/numerical audits remain outside
the production selection/stability algorithm. No historical calibration or default
changes, and no automatic policy selection, are introduced.


The [empirical curation layer](empirical-benchmark-foundation.md) reuses structural
validation and typed tabular ingestion. Source integrity, exact column roles and
sampling grain are core declarations; the workflow exports label-free manifests
and separate reference/audit files. A versioned protocol binds all profile roles
and future settings, and explicitly withholds fit authorization. This separates
syntactic ingestion readiness from scientific applicability without weakening
any existing model assumption. No empirical evaluator or estimator is added.


The [empirical evidence assessment](empirical-evidence-gates.md) is separate from
schema-1 curation and all estimators. An immutable review binds source, profile,
protocol and evidence snapshots. The workflow reconstructs curated semantics,
keeps computed facts separate from reviewed claims, and derives purpose-specific
blocked/inapplicable/available decisions without a fitting dispatch. No readiness
Boolean can override an existing model's independence requirement.


The [replacement source audit](replacement-empirical-audit.md) is a separate
core declaration, narrow source-inspection adapter and orchestration workflow.
It reuses existing structural validation, pure table/landmark parsers and the
unchanged fixed-TPS reader for derivatives. Extended source TPS metadata is
retained outside analytical features; curve sections are audited without being
admitted to fixed-landmark models. Reference namespaces remain candidate-only,
and no fitting, model configuration or older empirical schema is changed.


The [correspondence assessment](skate-correspondence.md) adds separate immutable
evidence declarations, documentary CSV/official haul adapters and fit-free
orchestration. It binds the exact prior source audit, preserves candidate versus
documented links, and keeps correspondence coverage separate from all analytical
readiness. No old empirical schema, model or source reader is weakened.

The [amphipod source audit](amphipod-audit.md) adds a separate immutable plan,
bounded standard-library workbook adapter and fit-free workflow. Literal voucher,
molecular and raw locality context evidence remain distinct from analytical
readiness. Its typed raw measurements use existing tabular contracts without
changing production readers, models or older empirical schemas. Only locked
source metadata, invented engineering fixtures and compact findings are public.

The [amphipod evidence assessment](amphipod-evidence.md) binds and semantically
replays the original trait audit. Its separate streamed reference profile counts
formatting rows without turning them into specimens, preserves marker-donor and
partition provenance, and separates documentary links from independence-dependent
readiness. Existing readers, model paths and evidence schemas remain unchanged.


The [density family](tabular-density-clustering.md) has its own dependency-free
configuration, numerical adapter, artifact contract and CLI. It reuses existing
tabular validation/preprocessing without changing their GMM behavior. A generic
inclusion-conditional subsampling accumulator separates noise, positive groups,
coverage and failures; no absent-unit predictions or GMM significance evidence
are synthesized. The frozen paired study and independent numerical/denominator
auditor live in benchmarking modules, outside production inference. Existing
models, defaults, empirical gates and p-values remain preserved.


The [moons diagnostic](noisy-moons-diagnostics.md) lives entirely in benchmarking
modules with separate versioned configuration, source-archive reconstruction,
guarded hierarchy instrumentation, graph/branch diagnostics and read-only audits.
It reuses existing ingestion/PCA and the library estimator while leaving all
production fitting/sampling modules unchanged. Original-unit identities survive
archived draw replays. Synthetic fixture construction supports installed tests
without author contact or real empirical inputs. Frozen point arms and oracle
post-fit explanations remain separate from user-facing model selection.

The [unified tabular entry point](tabular-analysis.md) composes unchanged native
workflows using a frozen source, dependency-free arm configuration and a shared
report. Its label-invariant comparison lives in `core.partition_comparison`;
family adapters preserve native evidence rather than coercing density noise
into the GMM delimitation-result contract. No model-family ranking is performed.

The [grouped analysis adapter](grouped-analysis.md) freezes both sources and the
identity mapping, then invokes existing ordinary/predictive grouped engines.
Its dependency-free configuration preserves native support boundaries; shared
reporting and label-invariant comparison are reused from the tabular interface.

The fit-free amphipod pilot protocol/cohort-feasibility checkpoint is complete: four
frozen morphology rosters, independent raw-cell accounting and a non-executable
comparison design. Sampling-dependent fitting, resampling, significance and species
validation remain blocked; the next empirical analysis requires an explicit
dependence interpretation and sufficient documented reference coverage. See
[the findings](amphipod-pilot-results.md).

The [missing-data tabular interface](tabular-missing.md) preserves the complete-data
default and adds versioned policy/cohort accounting with per-training-fit
imputation. Its [validation](tabular-missing-results.md) documents greater
retention alongside distorted geometry and spurious Gaussian partitions; it
promotes no default or empirical inference gate.

### Observed-coordinate Gaussian-mixture baseline

`models.observed_config`, `observed_likelihood` and `observed_gmm` isolate the
point-only observed-coordinate likelihood, constrained conditional-moment EM and
multistart candidate selection. `workflows.observed_gmm` reuses typed ingestion
and existing eligibility accounting without routing through PCA.
`benchmarking.observed_audit` reconstructs densities and updates independently
of the fitting implementation. This opt-in workflow adds no significance test,
resampling adapter or empirical readiness claim. See [the contract](tabular-observed.md).


### Observed-coordinate recovery diagnostics

The [recovery diagnostic](observed-recovery.md) lives entirely in benchmarking
modules. Continuations call the unchanged EM, retain objective/trace anchors and
reconstruct separate extended-budget candidates. Independent audits reuse the
dense marginal/moment reference; known-generating-parameter starts remain
isolated from unsupervised selection. No production configuration, default,
old serialization, empirical gate or historical p-value is changed.

### Fixed-corruption tabular representation

The opt-in [denoising adapter](tabular-denoising.md) has a separate dependency-free
configuration and trainer. It reuses the ordinary fixed network, numeric forward
transform, density fitter and inclusion-conditional accumulator while preserving
the legacy numerical modules. A shared workflow helper owns ingestion, artifact
exports, reports and strict numeric loading for both neural families. Corruption
has its own local CPU generator; checkpoints and noise digests support a separate
NumPy gradient/Adam auditor. The [paired comparison](denoising-study.md) freezes
feature-map/noise controls, all neural seeds and resource gates. Production
defaults, old serialization, p-values and empirical evidence gates stay unchanged.


### Gaussian-reference graph-count feasibility

The versioned `benchmarking.gaussian_graph_*` modules own the frozen synthetic
grid, PSD-aware raw Gaussian reference, independent-pilot normalization, paired
count decisions, local evidence, resource gates and independent regeneration
audits. They reuse the existing graph/assignment primitives and unchanged
baseline workflows; production source and schema behavior remain intact.
The entry point is a benchmark module, not a new `cluster-*` command. The
[feasibility study](gaussian-reference-spectral-results.md) does not support
promotion: fewer Gaussian splits coexist with missed nonlinear groups.

### Matched neural integration

The separate `models.matched_*` modules own strict paired identity alignment,
post-standardization view weights and frozen paired transforms. The new workflow
uses the unchanged ordinary CPU trainer, latent density fitter and native
inclusion-conditional support accumulator. Whole-individual draws select both
measurements together and refit every learned transformation. Other modalities
can remain unselected; population/locality ecology is not broadcast to individuals.
The [input and output contract](matched-neural.md) deliberately excludes grouped,
unmatched and incomplete neural inputs. Numeric encoder/decoder reloads need no
PyTorch and do not predict HDBSCAN memberships for new individuals.

The `benchmarking.matched_*` modules isolate synthetic laws, three-seed
comparisons, retained read-only training traces and independent NumPy optimizer
audits. Generating labels are evaluation-only. Loss balance does not establish
equal biological influence, and native five-draw pilot summaries do not validate
species supports. No existing trainer, model configuration or p-value changes.

The approved [numerical-audit amendment](matched-neural-audit-amendment.md)
isolates read-only optimizer observers from user fitting. Independent NumPy
gradients, moments and updates check each actual training step, with exact
original-weight/loss replay and retained source/input hashes. The original
serial-rollout guard failures remain evidence; the auditor does not select
candidates or alter training. [Results and limits](matched-neural-results.md)
keep computational verification separate from group recovery.

## Archival matched-view diagnostics

`benchmarking/view_signal_*` separates source binding, frozen-coordinate
replay, post-fit geometry, density fitting, independent auditing and reports.
The opt-in `diagnose-matched-view-signal` command never enters training or GMM
fitting. It reuses the unchanged density core and guards native assignment
parity. Frozen source/runtime/settings, audited preflight and measured resources
are required before its main archival analysis. Invented engineering fixtures
are explicitly separate from scientific evidence. Production defaults,
calibration workflows and empirical gates are unaffected.


### Separate matched-view encoders

`models.separate_*` owns a distinct, opt-in schema and two unchanged native
encoders. Each branch matches its standalone fit exactly; standardized branch
coordinates are concatenated without joint compression. The shared paired
workflow helper owns identity-bound ingestion, exports and failure handling;
legacy two-coordinate output and serialization remain compatible. A separate
paired resampler refits both branches on the same whole-individual draws.

The `benchmarking.separate_*` modules freeze architecture/dimension controls,
independent raw laws and all neural seeds. Shared lifecycle, numerical observers,
rate summaries and explicit per-arm resource accounting avoid duplicated study
logic. Benchmark-only joint-four and per-view PCA controls cannot be selected
through the public clustering adapter. See the [input contract](separate-neural.md)
and [prespecified comparison](separate-neural-study.md).


### Frozen-coordinate clustering diagnosis

The `benchmarking.coordinate_*` modules bind the completed separate-encoder
archive, reconstruct its exact frozen representations without Torch/SNPio,
and compare existing HDBSCAN and GMM algorithms on the same two/four dimensions.
The shared experiment lifecycle accepts explicit archival metadata/audit callbacks
while preserving legacy defaults. New numerical work stays in diagnostics; all
production model and sampling code remains byte-identical to the source checkpoint.

A read-only observer captures each attempted GMM candidate without changing
optimization or winning-start eligibility. Dense independent likelihood/BIC and
hierarchy audits distinguish fitting, admissibility and model selection.
Counterfactual tree policies and reference-label geometry are evaluation-only.
Public exports contain aggregate counts and hashes; per-fit records stay local.
See the [bounded protocol](coordinate-diagnosis.md).

### Separate encoders with a shared GMM

`models.separate_gmm` reuses both unchanged native branch trainers and fusion,
then uses the established tied-covariance BIC search. Its distinct configuration
and paired workflow reuse identity validation, numeric exports and native
matched subsampling. Frozen prediction requires numerical libraries without
Torch or SNPio; inference validates component weights, positive-definite shared
covariance and stored shapes/hashes. Numerical failures retain partial evidence
and withhold assignments. Existing models and serialization remain compatible.

The `benchmarking.separate_gmm_*` modules bind archived replays, freeze fresh
data/settings and reuse the shared experiment lifecycle with a measured resource
gate. A read-only observer audits every actual Adam update and every candidate
density. Heads share exact coordinates, while every subsample refits both
branches. Independent reporting checks identities, likelihoods, BIC, intervals
and coassignment denominators. Public exports contain cell aggregates; fits,
weights and trajectories remain local. See the [model contract](separate-gmm.md),
[protocol](separate-gmm-study.md) and [results](separate-gmm-results.md).

The opt-in [neural calibration adapter](neural-calibration.md) composes a strict
dependency-free configuration, the unchanged raw-Gaussian estimator, compact
native point fits and the existing lossless fitted-calibration engine. Both
encoders and the full shared-GMM search are refitted for every generated raw
table; point-array parity with the existing model is tested. Failed fits retain
their evidence and withhold incomplete inference. Existing point workflows,
default choices and historical p-values remain unchanged.

The `benchmarking.neural_calibration_*` modules separate simulation, immutable
source/seed freezing, paired neural/PCA fits, independent numerical auditing,
resource gates and aggregate reporting. The [initial pilot](neural-calibration-results.md)
completed with strong standard four-group recovery but an unresolved Gaussian
false split. The [larger follow-up](neural-false-splits-results.md) assesses two
Gaussian laws at N=64; empirical species claims, related sampling and calibration
after seed/model selection remain excluded.

The additive `benchmarking.neural_false_split_*` study layer reuses the unchanged
raw simulator, calibration engine and numerical auditors. It owns a separately
versioned design, disjoint seed registry, fixed balanced batches, per-unit evidence
seals and cumulative runtime/storage gates. Completed units are verified and
skipped; interrupted units retain unavailable inference and are not replaced.
Archive binding checks source and runtime hashes, and full calibration replays
compare scientific results independently of timing metadata. Aggregate reporting
keeps ordinary splits, global Gaussian rejection and the selected archived
precision diagnostic separate, with requested denominators and failure bounds.
See the [batched protocol](neural-false-splits.md).

### Paired calibrated power and group recovery

The additive `benchmarking.neural_power_*` layer specifies independent raw signal,
Gaussian-control and stress laws around the unchanged neural/PCA calibration
adapters. The reusable `SealedCalibrationStudy` receives explicit design,
job, worker and auditor dependencies; existing study engines remain unchanged.
It verifies sealed whole work units before continuation, retains failed guards
and interruptions, and charges cumulative serial execution and audit time.

Source-bound archival replay binding, disjoint seed accounting and a separately
measured preflight precede the main experiment. Reporting separates global
Gaussian departure, generating-partition recovery and their joint event, with
requested denominators, failure bounds and simultaneous uncertainty. Public
exports contain cell aggregates only. This layer adds an operating-range study,
not a new production model, adaptive configuration choice or species test. See
the [frozen protocol](neural-power.md).


### Archival weak-signal diagnosis

`benchmarking.weak_signal_*` binds original complete unit inventories and extracts
only observed raw/coordinate evidence. Its dependency-free contract and explicit
source-lifecycle dependencies reuse `SealedCalibrationStudy`; existing kernels
and study engines remain unchanged. Known Gaussian references, independent dense
and affine audits, exact native-head replays and isolated supervised truth starts
have separate roles. Scientific execution blocks Torch/SNPio imports, with no
representation retraining or historical calibration changes. Reporting preserves
paired dataset accounting and exports cell aggregates only. See the
[fixed diagnostic protocol](weak-signal.md).

### Supervised reference assignment and nonlinear measurement pilot

`models.residual_config` owns a separate dependency-free contract;
`models.residual_network` owns fixed CPU training and NumPy frozen prediction.
`io.reference_labels` validates a separate ID-bound label source. The workflow
uses established table ingestion/alignment and training-only equal-view scaling,
then exports reference predictions with novelty explicitly unassessed. Numeric
reload checks hashes, shapes, classes and fitted preprocessing without pickle.
It does not reuse the delimitation-result contract for supervised assignments.

The additive `benchmarking.residual_*` layer reuses `SealedCalibrationStudy`,
the unchanged native point-only PCA/neural/shared-GMM adapters and existing
actual-path autoencoder audits. Independent residual derivatives/Adam checks,
nonlinear measurement equations and external-cohort metrics are separate from
fitting. Allometry, saturation, nuisance shifts and an unfamiliar reference class
have frozen seeds/settings. Supervised and discovery tasks stay distinct; all
three seeds are retained and no model ranking changes defaults or p-values.
See the [reference interface](residual-reference.md) and
[bounded protocol](residual-nonlinear.md).

### Opt-in reference support and abstention

`models.reference_abstention_config` owns the dependency-free, separately versioned
policy contract. `models.reference_abstention` separates training-only raw support,
classwise held-out ranks and the four decision states. `reference_projection`
reuses strict matched-table validation with linear-memory cohort projection.
The numerical classifier trainers and original probabilities remain unchanged.

The additive workflow binds original model/training identities and a separate
labelled calibration cohort, validates numeric artifacts and copied labels, and
writes completion only after required artifacts succeed. Frozen loading/inference
needs NumPy, with no Torch or SNPio import. Legacy train/predict commands keep
their existing schemas and behavior; the new commands are explicitly opt-in.

`benchmarking.abstention_*` reuses the unchanged sealed lifecycle and residual
actual-update auditor. Independent raw-law, inventory, quadratic-score, rank,
decision and metric audits retain failures and separate known-class compatibility
from unfamiliar-group acceptance. Classwise and nuisance-shift evidence stays
explicit; neither a compatible set nor withholding certifies species status.
See the [interface](reference-abstention.md), [protocol](reference-abstention-validation.md)
and [results](reference-abstention-results.md).


### Opt-in class-local reference support

`models.local_reference_config` defines a separate versioned contract;
`models.local_reference` reuses the frozen pooled support metric and calculates
fifth-neighbor scores in bounded chunks. Original training IDs order neighbor
provenance; calibration IDs only set independent classwise ranks. The additive
workflow binds classifier, feature, label and source hashes and loads numeric
artifacts without Torch or pickle. Old policies and model probabilities stay intact.

`benchmarking.local_reference_*` binds sealed predecessor observations and reuses
the unchanged lifecycle and actual-update residual auditor. Independent direct
quadratic distances, neighbor/rank audits and analytical known-law density curves
are distinct from fitting. Reporting retains all neural seeds, classwise ambiguity,
nuisance/overlap controls and requested denominators. No evaluation label selects
a rule or default. See the [interface](local-reference-support.md),
[protocol](local-reference-validation.md) and [negative results](local-reference-results.md).
