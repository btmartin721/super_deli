# Super Deli Milestone 1 — One workflow for candidate species hypotheses

**Draft for implementation review · 17 November 2026 · baseline `c97b844`**

A researcher selects measurements, names the biological unit, and receives a plan showing which hypotheses the data can address. Given a completed analysis, the same Python package can show the candidate partition, the specimens assigned to it, and the evidence for a particular split. Milestone 1 specifies the common interface and its first two fit-free operations. All numerical results in this document come from completed work.

## 1. The 15 input combinations

**G** means SNPio genomic data, **S** means geometric morphometric shape, **E** means ecological measurements, and **V** means environmental measurements. E describes the organism's ecology, such as individual diet or microhabitat use. V describes its surroundings, such as site salinity or a climate-cell average. A name such as `temperature` or `environment` cannot establish the measurement level or its analytical role. Conventional morphological traits such as counts and linear measurements retain their existing `morphological` kind as an additional optional input outside this four-modality matrix.

Status describes **native capability at the baseline commit**. *Implemented* means an existing workflow accepts the stated design. *Partial* means some numerical analyses are possible if V occupies the single legacy `ecological` slot. *Planned* means the requested inputs cannot currently enter the same model as distinct views. The common interface below is proposed for every row. A valid plan for a planned route explains the missing capability.

| Selected | Current status | Native route or missing capability | Unit and result the user can expect |
|---|---|---|---|
| G | Implemented | `baseline-genomic` using SNPio | Specimen genetic groups and conditional memberships. Locus/block stability when present. |
| S | Implemented | `cluster-shape` on fixed homologous landmarks | Candidate shape groups at the declared unit, generally specimens. |
| E | Implemented | `cluster-tabular` on numeric ecological features | Ecological groups of the measured specimens, populations, or localities. |
| V | Partial | Numeric environment can occupy one legacy ecology table in `cluster-tabular`. First-class V is new. | Environmental groups of sampled units. Site-only measurements resolve sites. |
| G + S | Implemented | `analyze-matched-multimodal`; grouped designs can use `cluster-integrative`. | Matched specimen partitions, or an explicitly grouped target partition. |
| G + E | Implemented | `analyze-matched-genomic`; grouped designs can use `cluster-integrative`. | Individual ecology and SNPs on matched specimens, or a partition at the common grouped level. |
| G + V | Partial | `cluster-integrative` can use G and a legacy ecology slot for V. | For site V, link genotyped specimens to sites and aggregate their genomic scores explicitly. The result assigns sites. |
| S + E | Implemented | `analyze-matched-multimodal`; grouped designs can use `cluster-integrative`. | Shape and individual ecology on matched specimens, or a grouped partition with declared links. |
| S + V | Partial | `cluster-integrative` can use S and a legacy ecology slot for V. | For site V, aggregate linked specimen shape scores after shape preprocessing. The result assigns sites. |
| E + V | Planned | Two distinct tabular kinds and a method accepting both are needed. | A common specimen or grouped target with explicit measurement links. |
| G + S + E | Implemented | `analyze-matched-multimodal`; `cluster-integrative` also accepts these three kinds. | Joint and singleton partitions on matched specimens where the coordinator applies. Grouped integration assigns its declared targets. |
| G + S + V | Partial | `cluster-integrative` can combine G, S, and V in the legacy ecology slot. | Common target profiles. Grouping to localities requires links from each specimen-level source. |
| G + E + V | Planned | The genomic route needs distinct E and V blocks. | Genomic, ecological, and environmental evidence at the declared common target. |
| S + E + V | Planned | The shape route needs distinct E and V blocks. | Shape, ecological, and environmental evidence at the declared common target. |
| G + S + E + V | Planned | Four distinct kinds including V and a compatible coordinator are needed. The existing four-view example uses conventional morphology as its fourth kind. | Integrated result plus comparable single-view results, with the role and sampling unit of each view visible. |

The seven implemented rows inherit their native restrictions. Matched coordinators require complete, explicit specimen links, supported diploid biallelic calls when G is selected, complete selected continuous tabular values, and reviewed coordinates when S is selected. Individual ecology must actually have been measured on the specimen. The lower-level integrator supports separate grouped and complete-case designs. It does not automatically produce same-cohort singleton fits. The plan and report must identify which of those products actually exist. [SNPio entry point](genomic-ingestion.md) · [Shape](shape-clustering.md) · [Tabular](tabular-clustering.md) · [Matched genomics](matched-genomic.md) · [Matched shape](matched-multimodal.md) · [Grouped and matched integration](integrative-clustering.md)

## 2. Data and inference contract

1. **Keep identity explicit.** Use the current dataset, observation, specimen, population and locality IDs. Join by declared IDs, preserve original observation IDs, record the target level, and expose excluded units with reasons. A SNPio sample links to a specimen. A popmap supplies metadata. A site record contributes once at its measurement level. Several photographs or repeated readings of one specimen remain linked observations.
2. **Give V a distinct declaration.** In the proposed next dataset schema, add `environmental` with the existing typed CSV/TSV parser. Retain an observation ID, sampled unit ID, source and measurement units. Record spatial support (point/site or grid cell), the date or time window when applicable, and the coordinate reference system when coordinates locate the observation. Record shared source cell or sampling event identities when several units share a measurement. Feature roles already distinguish analytical variables from covariates. A covariate-only view requires an adjustment-capable method before it can enter a fit. Raster extraction and temporal matching are later adapters. The first input is an explicitly linked table.
3. **Read old declarations literally.** Keep schema-1 `ecological` declarations, configurations and hashes valid. A user who wants to identify a legacy table as V supplies a new versioned declaration with provenance. Do not infer its meaning from a filename, silently relabel it, combine E and V into a single block, or use the conventional morphology slot for either. Two views derived from the same measurements share provenance and cannot count as independent corroboration.
4. **Separate modality selection from incomplete coverage.** Any nonempty subset is a valid *request*. For specimen-level matched analysis, require the selected observations to cover the same declared specimen cohort. An explicitly requested native complete-case route records the intersection and exclusions. For grouped integration, require the existing mapping and aggregation policy per view. A site value cannot become independently replicated evidence for every specimen there. With no common target or defensible links, return the reason and the feasible standalone routes. Estimating missing views and assigning sympatric specimens from site-only measurements remain outside this baseline.
5. **Preserve evidence scope.** Distinguish specimen membership conditional on a selected model, stability across resamples, model comparison within one search, and evidence about an evolutionary boundary. Compare partitions through shared units and pair membership. Cluster labels have meaning within their own fit. Raw images and eventual LiDAR records feed S through reviewed landmark measurements. The fit-free planner can identify that prerequisite for an image-only declaration.

Extend the reporting layer around the existing data and result classes. Retain signed predictive gains, count denominators and availability in typed report fields. The legacy `Evidence` type restricts scalar probability/frequency values to [0, 1], so it cannot represent a signed gain. [Data declarations](../src/super_deli/core/data.py) · [Result contracts](../src/super_deli/core/results.py) · [Integrative settings](../src/super_deli/models/integrative_config.py)

## 3. Proposed Python API and CLI

These names specify future interfaces. The examples use real input paths. Run the examples from the repository root once the interfaces exist.

```python
from pathlib import Path

from super_deli.analysis import plan_analysis, report_from_archive

plan = plan_analysis(
    Path("examples/integrative/matched_dataset.json"),
    modalities=("markers", "outline"),
    target_level="specimen",
    missing_views="error",
)
print(plan.to_dict())

study = Path.home() / ".codex" / "outputs" / "super-deli" / "matched-multimodal-20261114"
report = report_from_archive(
    study / "validation" / "main" / "02-complementary-analysis",
    study_manifest=study / "study-manifest.json",
    output_dir=Path("outputs/m1-rendered-report"),
)
print(report.html_path)
```

```bash
super-deli analyze plan examples/integrative/matched_dataset.json \
  --modality markers --modality outline --target-level specimen --missing-views error

study="$HOME/.codex/outputs/super-deli/matched-multimodal-20261114"
super-deli analyze report "$study/validation/main/02-complementary-analysis" \
  --study-manifest "$study/study-manifest.json" --output-dir outputs/m1-rendered-report
```

The Python and CLI report calls are equivalent alternatives, each requiring a fresh output directory. The CLI passes arguments to the same Python implementation. Reserve an explicit `analyze run` operation for future, separately approved execution with a dataset and native model settings. Neither proposed operation invokes it.

Select modalities by their names in the dataset. The proposed `target_level` accepts `specimen`, `population`, or `locality`. The coverage choices are `error` and `complete_cases` where the native route supports them. For a grouped design, an optional per-view mapping such as `aggregation={"markers": "mean", "outline": "mean"}` corresponds to repeated CLI options `--aggregate markers=mean --aggregate outline=mean`. A source already at the target level uses `none`. If aggregation is necessary and unspecified, return a conditional plan describing the choice. Report when a grouped or complete-case request selects the lower-level integrator and therefore lacks same-cohort singleton comparisons.

| Operation | Reads and checks | Output and failure behavior |
|---|---|---|
| `plan_analysis` / `analyze plan` | Dataset declaration, selected modality names and local file references. Check unique names, kinds, declared coverage, unit links and route prerequisites. Use schema order for deterministic output. Genomic planning does not instantiate SNPio or parse genotypes. | Return a versioned plan; CLI emits JSON to stdout. Include dataset identity, selected views, target level, counts of declared, linked and excluded units, effective coverage policy, eligible native methods and outstanding checks. Distinguish declared coverage from verified measurement completeness. |
| `report_from_archive` / `analyze report` | A completed run from a recognized archive schema. Match configuration, cohort IDs and artifact digests. For sealed studies, verify each consumed input against the supplied study manifest. The first reader supports the archived matched-multimodal format used below. | Produce `report.json` and offline `report.html` from the same data in a fresh directory. Return their paths. Preserve native values and attach source artifact references. A failed verification, unsupported archive format or existing output directory yields a diagnostic and no completed report. |

A valid planning request returns `ready`, `conditional`, or `unsupported` plus reasons. `ready` means the declared design has a native route; file-content and numerical readiness can remain unchecked. `conditional` identifies a missing user choice or link a supported design requires. `unsupported` identifies a method or first-class view the implementation lacks. A proposed schema-2 V declaration is recognized by the new planner even while its numerical route remains unsupported. Invalid names, inconsistent IDs and malformed schemas are input errors. CLI exit 0 means a plan or report was produced, including a plan that explains an unsupported request. Invalid input or failure to produce the requested product exits 2, with the diagnostic on stderr.

For the checked-in plan example, the expected declared intersection is **128 of 128 specimens** for `markers` (PHYLIP) and `outline` (landmark CSV). The route is the matched multimodal coordinator with singleton G and S and a combined arm, subject to its native measurement checks. The other declared views do not affect this cohort. [Example dataset](../examples/integrative/matched_dataset.json)

## 4. Report content and archived example

| Report section | Required content |
|---|---|
| Run and cohort | Source workflow, dataset/configuration identity, biological unit, views and roles, membership of the comparison cohort, exclusions, and which sources were verified. |
| Partitions | Singleton and combined arms, resolved counts, selected K, search-limit flags, and any failed or missing arms. Only compare partitions on identified common units; give the denominator. |
| Specimen | Source IDs, arm-specific assignment, conditional membership and competing assignment when saved. Report abstentions, failures and unmeasured views explicitly. |
| Focal boundary | Identify two groups by originating arm and membership sets. For each comparable arm, count cross-group specimen pairs separated and within-group pairs kept together, each with its own denominator. A partition could separate both groups while fragmenting each one. Give group-recurrence counts and resampling unit if archived. Descriptive same-cohort agreement, a separately fitted merge alternative, and independent validation each have their own field. |
| Uncertainty and provenance | Give requested, successful, failed and covered denominators as appropriate. Mark evidence `available`, `not_computed`, or `unavailable` with a reason, retaining exact unrounded values when available. Keep historic p-values with their original null and scope. Experimental held-out gain belongs to its own dataset-level field. |

**Portable example from the matched multimodal study.** Each column below summarizes four independent synthetic datasets. Each entry is the K selected in all four datasets in that column. The combined analysis used G, S, individual E and conventional morphology. V was absent.

| Archived arm | Complementary genomic and shape signals | Shape-only signal |
|---|---:|---:|
| Genomic | 2 | 1 |
| Shape | 2 | 4 |
| Individual ecology | 1 | 1 |
| Conventional morphology | 1 | 1 |
| Combined | 4 | 1 |

The combined fit recovered complementary structure in the first design and lost the shape partition in the second. These cells illustrate why the report shows the constituent arms. They provide no rate estimate for biological species discovery. [Published aggregate JSON](../examples/multimodal/results/validation.json) · [Study design and limitations](matched-multimodal-results.md)

**Local specimen-and-boundary example.** The companion `outputs/milestone-1/archived-example-report.md` in the planning worktree uses one completed complementary-signal dataset. For two 16-member combined groups, shape separated 256/256 cross-group pairs and genomics separated 0/256. Both exact groups recurred together in 2/3 paired bootstrap draws. The report traces two specimens across arms, displays the least decisive combined assignment, and identifies unavailable boundary evidence. The ignored output location keeps detailed native records local. An implementer with the archive can reproduce the example from the run and manifest paths in §3.

**Separate rare-group benchmark.** In the held-out study, 20/20 rare-six datasets had positive predictive gain and 0/20 met correct point recovery. Among draws retaining at least nine of ten rare-group members, 352/353 recovered at N=128 and 256 pooled, and 140/174 recovered at N=64. The latter missed 27 times at component-size eligibility and seven times at BIC selection. These are repeated draws within datasets. They cannot populate a specimen, split or merge support field in the complementary example. The historical p-values and diagnostic defaults remain those of their source workflows. [Held-out study and audit](heldout-support-results.md)

## 5. Handoff for GPT-6.2 Sol

**The first implementation ticket covers the declaration planner and the archived-report reader.** Begin it after implementation approval. Start in the existing `codex/heldout-support` worktree, check its current HEAD and working changes, and preserve this draft and the closed checkpoint. Use the current SNPio adapter for any future genomic execution. The fit-free operations need only declarations and saved artifacts.

1. Implement one route resolver shared by Python and CLI. Start with the legacy [validation rules](../src/super_deli/core/validation.py), [matched genomic settings](../src/super_deli/models/genomic_analysis_config.py) and [matched shape settings](../src/super_deli/models/multimodal_analysis_config.py). Accept a versioned environmental declaration for planning and expose its actual execution status. Keep v1 serialization and native settings intact.
2. Implement a read-only adapter for completed matched-multimodal runs. Use the linked [native report workflow](../src/super_deli/workflows/multimodal_analysis.py) to locate schemas, saved labels and probability columns. Read saved files directly. Even the existing `assess-integrative-support` command replays K=1 fits and is outside this ticket. Derive display counts using ID joins and label-invariant comparisons.
3. Verify all 15 subsets, Python/CLI equivalence, reordered IDs, grouped links, missing views, ecology plus environment, covariate-only environment and attempted site-to-specimen broadcasts. Include a valid legacy four-view request with conventional morphology. Use existing declarations, hand-authored fixtures and sealed outputs. Keep optional numerical packages out of planner imports.
4. Verify archival report parity on the §3 run. Exact source labels, K, memberships, denominators, provenance and unavailable fields must survive rendering. Check missing optional evidence, digest mismatches, an existing destination, and source immutability. Use fixture-only tests or static checks that perform no fit or resampling. Keep synthetic specimen-level output local.

Return the implemented routes and blocked designs, the fit-free test results, and a locally rendered version of the example for review. Opening a numerical execution route, evaluating a new model, CNN training, LiDAR calibration and running any fit require a separate decision. The held-out checkpoint is closed.
