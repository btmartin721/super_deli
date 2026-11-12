# Super Deli

Super Deli is being redesigned as a modular Python package for integrative
species-delimitation research using genomic, morphological, ecological, and
geometric morphometric data.

**Current checkpoint: supervised residual reference assignment and nonlinear benchmark.**
The [opt-in classifier](docs/residual-reference.md) trains on separately declared
reference labels and predicts complete matched morphology/ecology with frozen
numeric weights. It assigns known reference classes; novelty remains unassessed.
The [bounded nonlinear pilot](docs/residual-nonlinear-results.md) found no consistent
advantage over logistic regression, and both existing discovery pipelines remained
weak on allometric group recovery despite strong label-informed prediction.
Nuisance shifts reduced accuracy; unfamiliar classes often received high reference
confidence. All 700 main/preflight workflows and numerical audits completed.
Defaults, p-values and empirical gates remain unchanged; detailed evidence stays local.

**Previous checkpoint: weak-signal recovery diagnosis.**
The [archival diagnostic](docs/weak-signal-results.md) finds substantial generating
group overlap: the true-parameter raw-data classifier has about 86.9% expected
individual accuracy, and none of twenty weak datasets reaches the strict
four-group/ARI criterion under that reference. PCA preserves nearly the same
known-law accuracy. BIC favors fewer components; fixed-K and truth-informed
fits do not restore weak recovery. One isolated standard-signal PCA case shows
an optimization limitation. These are diagnostic findings, not formal ARI
ceilings or species guarantees. The [opt-in interface](docs/weak-signal.md)
reuses frozen data/coordinates; all 228 component fits and audits passed without
new neural training or calibration draws. Defaults and p-values remain unchanged;
detailed evidence stays local.

**Previous checkpoint: calibrated neural power and group recovery.**
The [completed paired study](docs/neural-power-results.md) tested 100 fresh
matched morphology/ecology datasets. Standard and strong four-group signals
were recovered well, while weak signals often rejected a Gaussian reference
without recovering all four groups. The PCA/shared-GMM control performed
similarly; the small cells establish no neural advantage. All 20,400 main
searches and a 2,200-search preflight passed numerical audits without failed
pipeline points. The [bounded study interface](docs/neural-power.md) preserves
completed work and records its approved, exact-parity runtime repair. Defaults,
historical p-values and empirical gates are unchanged; detailed evidence stays local.

**Previous checkpoint: larger Gaussian false-split validation.**
The [completed follow-up](docs/neural-false-splits-results.md) tested 400 fresh
N=64 Gaussian datasets with paired neural/PCA full-pipeline calibration. Neural
global rejection was 4.5% for independent views and 3.0% for correlated views,
with broad simultaneous uncertainty; this establishes no general size or species
guarantee. All 82,400 main searches and the 2,800-search preflight passed independent
audits without failed pipeline fits. The [batched study interface](docs/neural-false-splits.md)
preserves completed work, retains interruptions and enforces cumulative resource
limits. Detailed evidence stays local; defaults and historical p-values are unchanged.

**Previous checkpoint: full-pipeline neural/shared-GMM calibration.**
The separate opt-in [CLI/API](docs/neural-calibration.md) fits one joint Gaussian
model to complete matched raw morphology/ecology and regenerates both views for
99 full-pipeline refits, including neural training and the complete GMM search.
The [initial validation](docs/neural-calibration-results.md) completed 8,008 main
searches and an 800-search preflight without failures. Four-group recovery was
strong, but one of ten independent Gaussian controls still split and rejected;
ten datasets per law cannot establish a reliable false-positive rate. These are
approximate Gaussian-model departure tests, not species or selected-K tests.
Defaults and historical p-values are unchanged; detailed evidence stays local.

**Previous checkpoint: separate encoders with shared-GMM clustering.**
The opt-in [CLI/API](docs/separate-gmm.md) aligns complete individual morphology
and ecology, fits the unchanged encoder for each view, and selects shared-GMM
components by BIC on their four combined coordinates. Frozen transforms,
conditional memberships and matched whole-individual subsampling are supported.
The [fresh validation](docs/separate-gmm-results.md) used 520 datasets, all three
neural seeds and 7,516 main evaluations after a 568-evaluation preflight. Standard
four-group recovery improved, including N=64, but some Gaussian controls split;
unequal-spread small samples and curved moons remain limitations. Per-view PCA
with shared GMM also recovered the standard signals: a neural advantage is not
established. No default, historical p-value or empirical evidence gate changed;
detailed inputs, fits and training traces stay local.

**Previous checkpoint: frozen neural coordinate diagnosis.**
The [archival CLI/API](docs/coordinate-diagnosis.md) compares unchanged density
and full/shared Gaussian-mixture clustering on exactly the same saved coordinates,
without neural retraining. The [audited findings](docs/coordinate-diagnosis-results.md)
show recoverable complementary-group signal: shared GMMs recover all five archived
datasets per sample size and seed, while HDBSCAN merges them. Full GMMs lose
small-sample groups to BIC costs. Shared GMMs also split one small Gaussian control;
curved-moons recovery remains weak. This retrospective diagnosis nominates fresh
validation, not a new default or species decision. All 2,400 main evaluations and
480 preflight evaluations passed; detailed evidence stays local. Defaults and
historical p-values remain unchanged.

**Previous checkpoint: separate encoders for matched morphology/ecology.**
The opt-in [CLI/API](docs/separate-neural.md) fits one unchanged two-coordinate
autoencoder per view and applies HDBSCAN to their four concatenated coordinates.
It supports frozen transforms and matched whole-individual subsampling. The
[audited comparison](docs/separate-neural-results.md) retains every neural seed,
includes dimension/capacity controls, and completed 1,480 main workflows plus a
380-workflow preflight. Existing models, defaults, serialization and p-values
remain unchanged. No seed, family or species partition is automatically selected;
detailed inputs, weights and training traces stay local. Redundant groups were
recovered, but complementary and one-view signal recovery did not improve.

**Previous checkpoint: prespecified matched neural view weights.**
The opt-in [schema-2 CLI/API](docs/matched-view-weights.md) supports declared
morphology/ecology reconstruction fractions while preserving the legacy schema,
default, trainer and p-values. The [frozen study](docs/matched-view-weights-results.md)
completed 1,080 main workflows and a 300-workflow preflight, including twelve exact
legacy replays. Favoring a signal-bearing view improved recovery when its partner
was uninformative, but both unequal weights lost complementary four-group recovery.
No universal weight, seed or new default is selected. Detailed evidence stays local.

**Previous checkpoint: matched-view signal diagnosis.**
The [completed archival diagnostic](docs/matched-view-signal-results.md) traces
60 saved datasets through 21 fixed representations without neural retraining.
Uninformative ecology changes useful morphology distance structure before
compression; joint PCA and neural compression add further losses. Latent
standardization alone does not explain the failure. All 1,260 main fits and
660 native replays passed independent checks. A separate
[diagnostic CLI/API](docs/matched-view-signal-study.md) retains all seeds,
noise and failures. No repair, default or p-value changed. The subsequent
view-weight study above tests this diagnosis on fresh simulations.

**Previous checkpoint: matched morphology–ecology neural integration.**
The separate opt-in [CLI/API](docs/matched-neural.md) aligns complete continuous
measurements from the same individuals, balances the two views in a shared
autoencoder, and proposes HDBSCAN groups with explicit noise. It supports frozen
paired transforms and matched whole-individual subsampling. The
[frozen study](docs/matched-neural-study.md) compares all three neural seeds
with unchanged single-view and integrative PCA baselines. The
[audited results](docs/matched-neural-results.md) show useful complementary-signal
recovery, losses with uninformative ecology, and weak moons recovery.
Defaults, historical
p-values and empirical evidence gates remain unchanged; this is an experimental
research workflow, not a validated species decision.

**Previous checkpoint: Gaussian-reference spectral group-count feasibility.**
The [frozen benchmark-only study](docs/gaussian-reference-spectral-results.md)
reduced splitting of single Gaussian controls but did not recover the moons
groups. At N=256, both reference arms returned a computational K=1 for all
moons tables, illustrating that no supported split does not establish one
population. The rule remains benchmarking-only; no production model, default,
schema, historical p-value or empirical gate changed.

**Previous checkpoint: experimental unknown-K spectral clustering and fresh validation.**
The [separate point CLI/API](docs/tabular-spectral.md) estimates a group count
from complete raw continuous traits using a frozen eigengap rule with explicit
abstentions. The [320-dataset fresh validation](docs/unknown-k-spectral-results.md)
found substantial overpartitioning of low-dimensional Gaussian controls and
no correct two-group moons returns in the tested cells. This rule is not ready
for automatic species delimitation. It remains opt-in and experimental;
existing defaults, schemas, p-values and empirical gates are unchanged.

**Previous checkpoint: spectral-graph diagnosis on frozen representations.**
The [separate archival diagnostic](docs/spectral-graphs.md) compares graphs across
saved inputs, PCA and all neural seeds without retraining. The
[audited findings](docs/spectral-graphs-results.md) show substantial moons
membership signal when two groups are supplied, especially at N=256 on clean
raw traits. Small samples, noise and disconnected graphs remain limitations.
Supplied K is diagnostic: automatic group-count selection and fresh validation
are still required. Defaults, p-values and empirical gates remain unchanged.

**Previous checkpoint: opt-in denoising autoencoder + HDBSCAN.**
The [separate neural workflow](docs/tabular-denoising.md) adds fixed input
corruption during training, reproducible numeric checkpoints and whole-unit
subsampling. The [paired simulation findings](docs/denoising-results.md) retain
strong four-group performance but do not repair curved-moons recovery. No
default, historical p-value or empirical evidence gate changes.

**Earlier checkpoint: curved-moons neural recovery diagnosis.**
The [bounded archival diagnostic](docs/neural-moons.md) separates input geometry,
neural representation, latent scaling and density-group selection using saved
weights and existing simulations. The [findings](docs/neural-moons-results.md)
show mixed upstream and selection effects, with no reliable default repair.
Production models, defaults and historical p-values remain unchanged.

**First neural model completed: opt-in tabular autoencoder + HDBSCAN.**
The first neural workflow learns a nonlinear two-dimensional representation of
complete continuous morphology/ecology, then proposes density groups with
explicit noise. It retains numeric weights, training traces and optional
whole-unit subsampling, without changing existing defaults or p-values.
See the [CLI/API and limitations](docs/tabular-neural.md),
[frozen comparison protocol](docs/tabular-neural-study.md) and
[audited results](docs/tabular-neural-results.md).
This is a transductive research model, not an anatomical image detector or a
validated species-boundary test. PyTorch is an optional dependency; ordinary
installations and classical workflows remain independent of it.

**Preceding checkpoint: observed-data GMM recovery and convergence diagnosis.**
The [bounded diagnostic](docs/observed-recovery.md) separates BIC selection cost,
convergence, initialization and measured-information limits using preserved
archival evidence and 60 fresh datasets. Extended-budget and known-parameter
fits remain separate from production results; defaults and p-values are unchanged.
See the [audited findings](docs/observed-recovery-results.md).

The opt-in [observed-coordinate GMM](docs/tabular-observed.md) fits incomplete
continuous tables directly without PCA or completed-value substitution. It
remains a point-only workflow; uncertainty, significance and empirical sampling
gates are not resolved by this diagnostic.

**Earlier checkpoint: noisy-moons density-recovery diagnosis.**
The frozen diagnostic separates density connectivity, branch pruning and cluster
selection while preserving production code, defaults and historical p-values.
PCA retained both input dimensions. Original-noise small samples usually merged;
low-noise larger samples often fragmented despite retaining perfect generating
branches. Parameter changes traded one failure mode for another. See the
[protocol and CLI](docs/noisy-moons-diagnostics.md) and
[completed findings](docs/noisy-moons-results.md). Detailed evidence remains local;
no new default, significance test or empirical species claim is introduced.

**Preceding checkpoint: opt-in continuous tabular PCA + HDBSCAN.**
A second clustering family supports complete continuous morphology/ecology,
explicit unassigned/noise observations and whole-unit subsampling stability.
Existing PCA/GMM defaults and p-values remain unchanged. See the
[model, scope and CLI](docs/tabular-density-clustering.md) and
[paired simulation findings](docs/tabular-density-results.md).
Density groups and membership strengths are species hypotheses, not calibrated
species decisions or species probabilities. Genomic, shape and multimodal density
adapters and deep learning remain later work. Detailed evidence stays local.

**Preceding checkpoint: amphipod sampling and reference-evidence gates.**
The fit-free assessment corroborates three additional voucher/COI pairs, retains
one conflicting claim and seven unresolved cases, and documents 117 raw molecular
partition links. Sampling independence and species-reference truth remain
unestablished. See the [method](docs/amphipod-evidence.md) and
[findings](docs/amphipod-evidence-results.md). Defaults and p-values remain
unchanged; full evidence stays local and nobody was contacted.
The preceding [amphipod audit](docs/amphipod-audit-results.md) and
[skate assessment](docs/skate-correspondence-results.md) remain unchanged.

The preceding [empirical evidence review](docs/empirical-evidence-gates.md)
retains the penguin dataset's explicit limitations and its unchanged protocol.
No empirical clustering has been run in these fit-free checkpoints.

**Preceding checkpoint: optional scaled fold sizes for shared predictive stability.**
The separately versioned policy scales each training fold's minimum independent
locality support to its distinct sampled exposure. It keeps the full-data minimum,
bootstrap multiplicities, original-ID folds, positive predictive-gain gate and AIC
ranking. The unchanged legacy rule remains the default. See the
[method and CLI](docs/scaled-fold-stability.md) and
[paired validation](docs/scaled-fold-stability-results.md). The 40-dataset study
uses 99 paired whole-locality draws per policy, with independent numerical and
archive audits. Own-point and common-legacy-point stability are reported
separately; detailed evidence stays local. This is a predictive stability option,
without new significance calibration or species-probability claims.

**Preceding checkpoint: diagnosis of N=64 predictive-bootstrap instability.**
The frozen paired diagnostic preserves production fitting, sampling, defaults and
historical p-values. It audits 40 previous observations and 3,960 retained draws,
then compares multiplicities, unique support and additional optimizer starts on
40 fresh datasets. Truth-initialized fits remain separate from selection.
See the [protocol](docs/bootstrap-instability.md) and
[results](docs/bootstrap-instability-results.md). Detailed evidence stays local;
the controls are descriptive and do not define a corrected bootstrap or species
support.

**Preceding checkpoint: whole-locality bootstrap stability for shared predictive AIC.**
The separate opt-in workflow resamples whole localities, preserves their members
and multiplicities, and refits the complete fixed-fold predictive screen. It
reports selected-K frequencies, cluster recovery/merging and both-out-of-bag
coverage with conditional Monte Carlo precision and explicit failure bounds.
See the [method and CLI](docs/predictive-stability.md) and
[bounded validation](docs/predictive-stability-results.md). Defaults and historical
p-values remain unchanged; detailed evidence stays local. These frequencies
measure conditional resampling stability, not species probabilities.

**Preceding checkpoint: shared-covariance predictive-screened grouped AIC.**
The optional shared family pools within-cluster covariance while retaining the
same whole-locality predictive screen. See the [method](docs/shared-predictive.md)
and [completed paired validation](docs/shared-predictive-results.md): 350 fresh
observations, 4,200 paired candidate searches and 25 exact six-search archived
replays, with independent audits and uncertainty reporting. At N=64, screened K=4 recovery was 17/25 and 25/25 with shared covariance,
versus 0/25 and 1/25 with full covariance at signals 3 and 4.5. Shared screening
selected extra components in 5/150 Gaussian controls versus 1/150 under full
screening, and label agreement was lower under unequal-variance stress.
Covariance family must be prespecified; full covariance/BIC remain defaults and historical p-values
are unchanged. Detailed evidence remains local.
The preceding [N=64 diagnosis](docs/predictive-recovery-results.md),
[predictive-screen validation](docs/predictive-aic-results.md) and
[power study](docs/grouped-power-recovery-results.md) remain available.

The preceding [larger false-positive validation](docs/grouped-false-positive-validation-results.md)
completed 2,500 datasets and 497,500 fits; fitted-null rejection was 4.4–5.4%.
These selection-control cells do not replace that significance study. General calibration
and species-boundary validity remain unestablished.
Genomic, numeric morphology/ecology and shape scores can be fused by exact unit
IDs or explicit population/locality aggregation, with prespecified view weights.
The package also supports individual-modality PCA/GMM, frozen-fit shape
projection, visible marker detection and multimodal ingestion.
Anatomical landmark learning and formal species-boundary tests remain later work.

The new `benchmark-integrative` command evaluates the existing integrative model
and declared view subsets against separate generating partitions. It includes
eight stress scenarios, matched/grouped sampling, explicit failure denominators,
and independent-dataset Monte Carlo standard errors. See the
[benchmark protocol](docs/integrative-benchmarking.md) and
[bounded smoke results](docs/integrative-benchmark-smoke.md).
The [larger study results](docs/integrative-study.md) now report 800 simulated
datasets, 2,400 point-fit attempts, explicit failures and audited bootstrap logs.
`analyze-integrative-study` audits completed scenario shards and exports compact
dataset, paired-comparison, event-rate and support-bin tables without refitting.

The separate `calibrate-integrative` command tests a single Gaussian against the
configured GMM search in frozen joint scores, using an approximate parametric
bootstrap. It preserves the original clustering run and makes no species-boundary
or cluster-specific significance claim. See the [method and validation protocol](docs/gaussian-score-calibration.md).

The [completed calibration study](docs/gaussian-score-calibration-results.md)
reports 400 independent score datasets with explicit null, power and stress interpretations.

The separate `calibrate-reference-pipeline` command regenerates all four raw
modalities and refits ingestion, preprocessing, GPA, PCA, aggregation, fusion
and the complete GMM search for each Monte Carlo draw. Its global test uses a
known continuous-population simulator with explicit assumptions. See the
[reference-law method](docs/full-pipeline-calibration.md) and
[100-dataset validation results](docs/full-pipeline-calibration-results.md).
The empirical extensions below cover specific declared nulls; general empirical
nulls and species-boundary significance remain future work.

The new `calibrate-empirical-integrative` command estimates a joint Gaussian
null from matched, complete continuous morphology/ecology before PCA and refits
the integrative pipeline for every bootstrap draw. Its p-values are approximate.
See the [method](docs/empirical-null-calibration.md) and
[validation results](docs/empirical-null-calibration-results.md).
`calibrate-grouped-integrative` additionally fits joint between-group and
within-morphology Gaussian covariance with unequal specimen counts and one ecology
row per population/locality. It preserves the observed sampling design through
full-pipeline bootstrap refits, rejects unsupported covariance fits and keeps
missing-test denominators explicit. See the [grouped method](docs/grouped-null-calibration.md) and
[100-dataset results with explicit withheld tests](docs/grouped-null-calibration-results.md).
Schema-2 grouped calibration adds positive-semidefinite between-group covariance,
bounded convergence checks and a versioned sampler while retaining legacy audits.
See the [boundary method](docs/grouped-covariance-boundaries.md) and
[175-dataset results](docs/grouped-covariance-boundaries-results.md).
Additional fitted nulls for genomic/count/shape data remain separate checkpoints.

## Installation

Use Python 3.11 or 3.12. From this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The foundation has no runtime dependencies. Install the genomic extra to enable
SNPio ingestion on Python 3.11 or 3.12:

```bash
python -m pip install -e '.[dev,genomics]'
```

## Validate the illustrative dataset

```bash
super-deli validate examples/foundation/dataset.json --hash-inputs
```

The example includes specimen measurements, repeated observations, partial
specimen coverage, and locality-level environmental observations. Values are
illustrative and are not an empirical species-delimitation benchmark.

The command prints a JSON validation report, coverage counts, and provenance.
Add `--report /path/to/new-report.json` to also save it. Existing files are never
overwritten. Relative source references resolve against the configuration's
directory, independently of the working directory. `python -m super_deli`
provides the same CLI.

Exit codes: `0` for valid declarations/references, `1` for consistency/reference
errors, and `2` for malformed configuration, I/O, or usage errors. A successful
report still warns that modality contents and model suitability are unchecked.

## Python API

```python
from pathlib import Path

from super_deli import load_config, validate_dataset

source = Path("examples/foundation/dataset.json").absolute()
config = load_config(source)
report = validate_dataset(config, base_dir=source.parent)
print(report.to_dict())
```

See [the architecture and configuration contract](docs/architecture.md) for
biological levels, feature types, optional popmaps, uncertainty reporting rules,
validation boundaries, and the next checkpoints.

## Load genomic data

```bash
super-deli ingest-genomic examples/genomic/dataset.json --output-dir /tmp/super-deli-genomic-example
```

Choose a new output directory for each invocation. The adapter accepts VCF,
PHYLIP, STRUCTURE, and GENEPOP through SNPio's readers, with `popmapfile=null`
when no external map is supplied. It stages private input copies, preserves
source sample order, checks sample/specimen links, and records parsed IUPAC
calls, their missing-call mask, and provenance. Original inputs are unchanged.

```python
from super_deli.io.genomic import ingest_genomic

source = Path("examples/genomic/dataset.json").absolute()
loaded = ingest_genomic(
    load_config(source),
    base_dir=source.parent,
    output_dir=Path("/tmp/super-deli-genomic-api-example"),
)
genotype_data = loaded.genotype_data  # SNPio GenotypeData object
```

See [genomic ingestion](docs/genomic-ingestion.md) for reader options, outputs,
population metadata, and the precise validation boundaries. The hand-authored
fixtures specify expected genotype calls; they are not species-delimitation
benchmarks.

## Estimate candidate genetic clusters

```bash
python -m pip install -e '.[dev,genomics,baseline]'
super-deli baseline-genomic examples/baseline/dataset.json \
  --model-config examples/baseline/model.json \
  --blocks examples/baseline/blocks.json \
  --output-dir /tmp/super-deli-baseline-example
```

The baseline uses diploid biallelic dosages, explicit missingness/scaling policies,
PCA, and BIC selection of diagonal Gaussian mixtures including K=1. Bootstrap
replicates refit the complete pipeline using declared independent blocks or an
explicit unlinked-locus assumption. Conditional memberships, pairwise
co-assignment, and exact-cluster recovery remain distinct. Failed replicates are
retained; incomplete bootstrap estimates are withheld and bounded.

See [the baseline method and API](docs/genomic-baseline.md) for assumptions,
configuration, artifacts, reproducible synthetic truth, and validation limits.
The synthetic example is a software check, not empirical species validation.

## Load morphology and ecology

```bash
super-deli ingest-tabular examples/tabular/dataset.json \
  --output-dir /tmp/super-deli-tabular-example
```

The base installation supports continuous measurements, counts, binary states,
ordered scores and categorical traits, including units and covariates. Source
observation IDs must match declarations; repeated measurements and specimen,
population or locality levels remain explicit. The reader preserves missingness
and category labels. Select tables with repeated `--modality NAME` options.

See [tabular ingestion](docs/tabular-ingestion.md) for column/missing-token
options, the Python API, typed JSON artifacts, validation and processing limits.
The hand-authored example provides value/link truth for ingestion checks.

## Load geometric morphometric coordinates

```bash
super-deli ingest-morphometric examples/morphometric/dataset.json \
  --output-dir /tmp/super-deli-landmark-example
```

The base installation accepts fixed-landmark TPS (`LM=`/`LM3=`) and long-form
coordinate CSV. Declare dimensions, landmark order, biological observation links
and missing-value rules. Readers preserve coordinates, partial missingness,
repeat observations and scale/image metadata. Scale factors are recorded without
rescaling. This coordinate adapter retains images as metadata references.

See [morphometric ingestion](docs/morphometric-ingestion.md) for the Python API,
supported format subset, coordinate correspondence, artifacts and limitations.

## Load images

```bash
python -m pip install -e '.[dev,images]'
super-deli ingest-images examples/images/dataset.json \
  --output-dir /tmp/super-deli-image-example
```

Image manifests link each selected page/frame to a declared observation and
biological unit. The reader keeps verified source copies, writes RGB/RGBA PNG
display derivatives, and records orientation, resolution, bit depth and hashes.
Multipage/multiframe inputs require explicit selection. Pixel/file-size budgets
and explicit HEIF bit-depth policies prevent silent resizing or quantization.

See [image ingestion](docs/image-ingestion.md) for the manifest/options contract,
Python API, source preservation, color/coordinate limits and synthetic codec truth.

## Detect visible landmark markers

```bash
python -m pip install -e '.[dev,images,landmarks]'
super-deli detect-landmarks examples/detection/dataset.json \
  --detector-config examples/detection/detector.json \
  --output-dir /tmp/super-deli-detected-landmarks
```

Declare named RGB/search-region rules and component-size limits. Detection leaves
missing, ambiguous and potentially clipped markers null. Separate overlays and
diagnostics support review; coordinate JSON/CSV and an ingestible configuration
preserve biological links. Every centroid remains unreviewed, and statistical
localization uncertainty is explicitly unavailable pending calibration.

See [landmark detection](docs/landmark-detection.md) for configuration, the API,
coordinate conventions, synthetic accuracy checks and method boundaries.

## Align landmarks and estimate shape axes

```bash
python -m pip install -e '.[dev,shape]'
super-deli shape-pca examples/shape/train_dataset.json \
  --model-config examples/shape/model.json --output-dir /tmp/super-deli-shape-fit
super-deli shape-pca examples/shape/projection_dataset.json \
  --model-config examples/shape/model.json --reference-run /tmp/super-deli-shape-fit \
  --output-dir /tmp/super-deli-shape-projection
```

GPA removes translation, uniform size and rotation from homologous 2D/3D fixed
landmarks. Reflections are forbidden by default. Missing landmarks and repeated
units require explicit policies. Tangent-space covariance PCA records numerical
rank, explained variance and geometric distortion; saved parameters project new
observations without updating the training fit. This is descriptive ordination.

See [shape analysis](docs/shape-pca.md) for equations, policies, artifacts, API,
analytic truth tests and the provisional check of the supplied TPS download.

## Cluster shape scores and assess stability

```bash
python -m pip install -e '.[dev,clustering]'
super-deli cluster-shape examples/shape/clustering/dataset.json \
  --model-config examples/shape/clustering/model.json \
  --output-dir /tmp/super-deli-shape-clusters
```

This fits candidate shape groups using BIC (including K=1) in a fixed PC
dimension. Bootstrap draws resample biological units and refit GPA, PCA and
mixture selection. Repeated digitizations require aggregation first. Conditional
membership, bootstrap frequency and failed-replicate bounds are separate;
species-boundary p-values remain uncomputed. See
[shape clustering](docs/shape-clustering.md) for assumptions, OOB denominators,
configuration, artifacts, API and bounded synthetic validation.

## Cluster numeric morphology or ecology

```bash
python -m pip install -e '.[dev,clustering]'
super-deli cluster-tabular examples/tabular/clustering/dataset.json \
  --model-config examples/tabular/clustering/morphology_model.json \
  --modality traits --output-dir /tmp/super-deli-morphology-clusters
super-deli cluster-tabular examples/tabular/clustering/dataset.json \
  --model-config examples/tabular/clustering/ecology_model.json \
  --modality environment --output-dir /tmp/super-deli-ecology-clusters
```

This baseline analyzes one table of continuous/count features per run, with
explicit selection, scaling, count transformation and missingness policies.
Whole-unit bootstrap draws refit preprocessing, PCA and BIC/GMM selection.
Outputs preserve specimen, population or locality levels. Categories require
a numeric subset; repeated units require aggregation before fitting. See
[tabular clustering](docs/tabular-clustering.md) for assumptions, API, artifacts
and limits on interpreting environmental groups as biological species.

## Integrate linked modalities

```bash
python -m pip install -e '.[dev,clustering,genomics]'
super-deli cluster-integrative examples/integrative/grouped_dataset.json \
  --model-config examples/integrative/grouped_model.json \
  --output-dir /tmp/super-deli-integrative-groups
super-deli cluster-integrative examples/integrative/matched_dataset.json \
  --model-config examples/integrative/matched_model.json \
  --output-dir /tmp/super-deli-integrative-individuals
```

Select two to four view kinds and an explicit target biological level. The
baseline normalizes modality score profiles, fits joint PCA/GMM, and resamples
target units jointly across all views. Unmatched individuals require explicit
group links; entire missing views require an explicit complete-case policy.
Source memberships, exclusions, weights and numerical parameters are exported.
See [integrative clustering](docs/integrative-clustering.md) for the method,
configuration, reconstruction and conditional uncertainty scope. Components
remain candidate structure; formal species significance is uncomputed.

## Development checks

```bash
python -m pytest -q
ruff check
ruff format --check
python -m build
```

The test suite includes the read-only image/TPS manifest utility.
GitHub Actions tests the foundation on Python 3.11 and 3.12, validates the example,
builds wheel/source distributions, and checks wheel installation outside the
source checkout, including tabular and coordinate ingestion without scientific dependencies.
A separate genomic matrix tests the real SNPio readers with
and without external popmaps, the numerical baseline, real image codecs, visible
marker detection, GPA/shape PCA, unimodal/integrative clustering, and installed CLIs.
Optional tests are skipped
when extras are absent. Ruff coverage is declared centrally in `pyproject.toml`.
Linting covers the new package, tests, and synthetic-example generator.

The active implementation lives under `src/super_deli/`. Obsolete imputation
code and personal datasets are archived outside the repository.
The standalone `image_tps_manifest.py` keeps its existing manifest schema and
metadata checks, with PDF/HEIC/HEIF extensions added to discovery.

The [genomic fitted-null checkpoint](docs/genomic-null-calibration.md) adds an
approximate full-pipeline pooled-HWE test for explicitly unlinked diploid SNPs,
with optional popmaps and preserved failed-draw accounting.
See the [completed genomic validation report](docs/genomic-null-calibration-results.md)
for the frozen 125-dataset study and its inferential limits.

The [linkage-aware genomic checkpoint](docs/linked-genomic-calibration.md) adds
a separate fitted empirical block-profile null for complete genotypes and
prespecified independent blocks.
See the [completed linkage-aware validation](docs/linked-genomic-calibration-results.md)
for the frozen study, paired outcomes and sparse-support limitations.

The [joint genomic–phenotypic fitted-null workflow](docs/joint-genomic-phenotypic-calibration.md)
combines empirical genotype-block profiles with conditional Gaussian regression
of matched continuous morphology/ecology on prespecified raw genomic summaries.
It reruns the complete integrative pipeline and reports approximate joint-law
departure p-values, with stability and species-boundary interpretation separate.
See the [completed 125-dataset validation](docs/joint-genomic-phenotypic-calibration-results.md) for outcomes and limitations.

The [four-group recovery diagnostics](docs/integrative-recovery-diagnostics.md)
audit the generating signal, PCA retention, mixture constraints, optimization
and BIC penalties in the constructed joint-study alternative. They preserve
all models and significance results; the frozen grid is diagnostic only.
See the [completed recovery report](docs/integrative-recovery-diagnostics-results.md)
for all 525 point fits, the separate optimization diagnostic and its limitations.

An opt-in [shared-covariance integrative model](docs/shared-covariance-clustering.md)
reduces mixture complexity while retaining correlated joint scores. The
full-covariance default and existing significance results are preserved.
See [repository layout and local data](docs/repository-layout.md) for the cleaned
source tree and external-data policy.
See the [completed paired validation](docs/shared-covariance-clustering-results.md)
for recovery gains, unequal-variance/cline limitations and audited bootstrap support.

Shared-covariance full-pipeline significance calibration is available for the
matched genomic–phenotypic empirical null. See the
[scope and frozen validation](docs/shared-covariance-calibration.md). The covariance
family must be fixed before analysis; the default remains full covariance.

The [completed shared-family calibration](docs/shared-covariance-calibration-results.md)
reports all 125 fresh tests, conditional Monte Carlo intervals and stress limitations.

Shared-covariance [grouped morphology–ecology calibration](docs/shared-grouped-calibration.md)
uses the boundary-capable schema-2 raw null, explicit group links and fixed unequal
specimen counts. Legacy schema 1 retains its full-family scope. The covariance
family must be selected before analysis; stability remains separate.

The [completed grouped shared-family study](docs/shared-grouped-calibration-results.md)
reports 175 fresh observations, full-pipeline calibration and uncertainty.

The frozen [grouped recovery diagnostic](docs/grouped-recovery-diagnostics.md)
traces morphology/ecology signal through source PCA, group aggregation, fusion
and joint PCA. The [completed diagnosis](docs/grouped-recovery-diagnostics-results.md)
reports 750 point attempts (700 complete, 50 retained numerical failures), 75 separate
truth-initialized diagnostics and 25 reproduced archived baselines. Defaults and
existing p-values remain unchanged; exploratory settings are not promoted.

An opt-in [group-mean morphology PCA policy](docs/grouped-pca.md) gives each linked
population/locality equal influence on preprocessing and PCA. It supports frozen
projection and whole-group stability. Significance is now available within the
[bounded schema-2 grouped workflow](docs/groupmean-calibration.md). The [completed four-arm study](docs/grouped-pca-results.md) reports all
700 point workflows and 1,000 stability refits. Four joint PCs improved recovery
at N=256; group-mean PCA added no consistent benefit in these laws. Defaults and
historical p-values remain unchanged.

[Group-mean PCA significance calibration](docs/groupmean-calibration.md) supports
complete continuous grouped morphology and ecology using the schema-2 raw
hierarchical Gaussian null and a prespecified shared-covariance GMM. Every raw draw
refits group means, preprocessing/PCA, integration and the full K search. This
opt-in approximate departure test preserves defaults and historical results.

The [completed group-mean calibration study](docs/groupmean-calibration-results.md)
reports all 175 observations, full-pipeline checks and uncertainty.

[Full-covariance group-mean PCA calibration](docs/full-groupmean-calibration.md)
uses the same schema-2 grouped raw null with a prespecified covariance per mixture
component. Both covariance families are supported; fix one before analysis.
Defaults and historical p-values are preserved.

The [completed full-covariance group-mean study](docs/full-groupmean-calibration-results.md)
reports all 175 observations, full-pipeline checks and uncertainty.


The [Gaussian-control AIC diagnostic](docs/aic-gaussian-diagnostics.md) examines
finite-sample component selection, unequal-count measurement variance, optimization
and independent prediction. It preserves production fitting, defaults and all
historical p-values; detailed evidence stays local.

The [completed Gaussian-control diagnosis](docs/aic-gaussian-diagnostics-results.md)
reports 1,250 verified searches and the audit of 2,550 archived observations,
with uncertainty and independent held-out prediction.

## Unified tabular comparison

The opt-in [unified tabular workflow](docs/tabular-analysis.md) compares
prespecified PCA/GMM and PCA/HDBSCAN arms on one frozen continuous morphology
or ecology table. It reports noise-aware agreement, coverage and separate
family-specific stability without selecting a winning model or adding p-values.

The [unified grouped interface](docs/grouped-analysis.md) compares existing
full/shared BIC, AIC and predictive-screened AIC workflows for explicit
morphology/ecology groups with unequal specimen counts. Native uncertainty
contracts and existing defaults remain intact; it selects no winning species partition.

The [fit-free amphipod pilot protocol](docs/amphipod-pilot.md) freezes raw morphology
rosters and reports cohort feasibility while retaining unresolved inference gates.

The [opt-in missing-data comparison](docs/tabular-missing.md) compares complete
cases with training-only mean filling and preserves explicit coverage, failure
and uncertainty limits. See [the sensitivity results](docs/tabular-missing-results.md).

The opt-in [observed-data GMM](docs/tabular-observed.md) fits incomplete continuous
traits directly in raw feature space, retaining conditional-moment EM diagnostics.
It provides point clusters and conditional memberships, without new p-values or
a species-boundary guarantee; existing workflows and defaults are preserved.


The [observed-data recovery diagnostic](docs/observed-recovery.md) separates
small-sample component-selection cost from convergence and measured-information
limits. It preserves original withheld searches and stores extended-budget and
known-parameter diagnostic fits separately; it promotes no new model or default.


The opt-in [separate morphology/ecology encoders](docs/separate-neural.md) retain
two coordinates from each matched individual-level view and cluster their four
combined coordinates with HDBSCAN. This adds no species p-values or automatic
model selection. The [bounded comparison](docs/separate-neural-study.md) includes
unchanged joint-two, dimension-matched joint-four, per-view PCA, single-view
neural and integrative GMM controls.
