# Genomic PCA/GMM baseline

This checkpoint estimates **candidate genetic clusters** from one genomic modality.
It provides a transparent baseline for later species-delimitation research.
Clusters, conditional memberships, and locus-bootstrap stability are different
quantities. None is a species posterior or a calibrated test of a species boundary.
Morphology/ecology readers are documented in [tabular ingestion](tabular-ingestion.md).
Landmark/image processing, empirical benchmarks, and integrative models remain
later checkpoints.

## Run the bounded synthetic example

Use Python 3.11 or 3.12:

```bash
python -m pip install -e '.[dev,genomics,baseline]'
super-deli baseline-genomic examples/baseline/dataset.json \
  --model-config examples/baseline/model.json \
  --blocks examples/baseline/blocks.json \
  --output-dir /tmp/super-deli-baseline-example
```

Choose a new directory. Input paths resolve against the dataset configuration's
parent. Model/block paths resolve from the CLI working directory. Inputs are
staged through the existing SNPio adapter; source files and existing outputs are
preserved. The same VCF, PHYLIP, STRUCTURE, and GENEPOP readers and optional
popmaps are available. Population/locality labels and `truth.json` are never
predictors or model-selection inputs.

The example contains 48 independent simulated diploid individuals, two generating
populations with allele frequencies 0.12/0.88, and 32 independent blocks, each
with three perfectly linked locus copies before independent missingness (1.5%).
Generating labels are in `truth.json`, separate from the analysis configuration.
They establish simulation truth conditional on this simple generator, not
empirical species truth. Regenerate into a **new directory** with:

```bash
python examples/baseline/generate_example.py /tmp/new-synthetic-baseline
```

The generator records its seed and NumPy version. The stored fixture is the
benchmark input; byte-identical regeneration across library versions is not promised.

## Explicit configuration and assumptions

The model JSON rejects duplicate/unknown fields, nonfinite values, and unsupported
versions. Required fields are `schema_version: 1`, `ploidy: 2`,
`missing_policy: "error"` or `"mean"`, and
`assume_independent_specimens: true`. The last declaration is a scientific
assumption, not a check for relatedness. Each source sample must link to a unique
specimen; repeated specimens fail until an aggregation policy is implemented.
Related individuals, technical replicates, biased sampling, admixture, and
continuous isolation by distance require interpretation and additional validation.

| Setting | Default | Meaning |
| --- | --- | --- |
| `n_pcs` | 2 | Fixed PCA dimension; no dimension search |
| `max_k` | 4 | Search every integer K from 1 through this limit |
| `min_cluster_size` | 3 | Reject a fit with any smaller hard-assigned component (minimum 2) |
| `scaling` | `patterson` | `center`, `standard`, or `patterson` |
| `n_bootstrap` | 100 | Whole-block refits; zero explicitly disables stability |
| `assume_unlinked` | false | Explicit independent-locus assumption instead of a block map |
| `n_init` | 10 | Independent GMM initializations for each K |
| `max_iter`, `tol` | 500, 0.001 | EM convergence controls |
| `reg_covar` | 0.000001 | Positive covariance regularization |
| `assignment_threshold` | 0 | Report unresolved assignments below this conditional membership |
| `random_seed` | 0 | Model and resampling seed; overrides no dataset settings |
| `n_threads` | 1 | Scoped native numerical-library thread limit |
| `max_pairwise_bytes` | 268435456 | Budget for the estimated dense pairwise working arrays |

For stability, provide `--blocks` **or** set `assume_unlinked: true`.
Providing both is an error; linkage is never silently inferred. The block file is:

```json
{"schema_version": 1, "block_ids": ["block_a", "block_a", "block_b", "block_b"]}
```

There must be one nonempty block ID per **original source locus**, including
excluded loci, in source order, and at least two independent blocks. This tiny
schema example is not a sufficient biological study design. Repeated IDs need
not be contiguous. Users must choose blocks appropriate to recombination and LD;
two blocks permit execution but cannot establish reliable bootstrap approximation.
Neither chromosomes nor fixed windows automatically establish independence.

## Encoding and point fit

1. Use `SNPio.GenotypeEncoder.genotypes_012` on an isolated `GenotypeData.copy()`.
   Seed its local reference-allele tie-breaking generator. Preserve source
   sample order and all original locus positions. Allele reversal at a locus
   preserves centered/scaled distances. Save the actual encoded matrix.
2. Exclude whole loci with more than two observed alleles or unsupported diploid
   IUPAC symbols. VCF additionally excludes multiallelic/non-SNP metadata, checks
   original called GT ploidy, and masks partial calls as missing. VCF records
   must correspond one-to-one to SNPio loci. Non-VCF diploidy relies on the user's
   declaration because IUPAC cannot reveal original ploidy. Other SNPio reader
   representation limits still apply; genotype likelihoods are not modeled.
3. Remove all-missing loci and loci with zero observed dosage variance. Reject
   a specimen with no observed retained calls, too few variable loci, or a
   rank-deficient requested PCA space. Report exclusions with zero-based indices.
4. Compute each retained locus mean from observed calls. `error` rejects any
   remaining missing calls. `mean` imputes with that locus's observed mean.
   Center dosages, then divide by 1 (`center`), the observed-call population
   standard deviation (`standard`, denominator = number observed), or
   `sqrt(2*p*(1-p))` (`patterson`, `p = observed mean/2`). Mean imputation reduces
   variance and can cause artifacts under structured missingness; it does not
   propagate genotype uncertainty or correct missingness bias.
5. Fit deterministic full-SVD PCA without whitening. Fit diagonal-covariance
   Gaussian mixtures on exactly these same coordinates for all candidate K.
   Fixed PCA dimension and the diagonal approximation are modeling choices.
6. Select the lowest BIC among admissible fits; ties favor smaller K. BIC is
   `-2*log_likelihood + parameter_count*log(n_specimens)` as computed by
   scikit-learn. K=1 is always considered. Insufficient-sample candidates and
   converged fits rejected by the declared hard cluster-size rule remain in the
   candidate table. The size rule filters the best EM fit for each K; it is not
   a constrained-likelihood optimizer. Any numerical/convergence failure of a
   feasible K fails the entire fit rather than changing the candidate search.

BIC is an approximate model-selection criterion for mixtures with regularized
covariances, not a significance test, and cannot be compared across different
PCA dimensions, resampled datasets, or feature spaces here. The selected K depends
on the candidate range, sampling and modeling choices. The summary flags a
selection at the upper search limit. Check these sensitivities before biological
interpretation. No automatic hyperparameter tuning or formal null calibration is
implemented. See the official [GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
and [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
API documentation for estimator definitions.

## Block-bootstrap stability

Each replicate draws the declared number of whole blocks uniformly with
replacement. Every locus in each drawn block is retained, including repeated
draws. Unequal block sizes produce varying locus counts. Empty/uninformative
blocks are not silently removed from the resampling frame. Each replicate refits
exclusions, imputation, scaling, PCA, all candidate K, and BIC selection on the
same sampled specimens. These are i.i.d. block-resampling assumptions; arbitrary
dependent blocks do not acquire valid uncertainty by being listed in JSON.

Replicate seeds are spawned independently from a fixed root seed. Changing the
number of bootstrap replicates leaves the point-fit seed unchanged. Labels are
canonicalized by first specimen occurrence; stability is label-invariant:

- **Pairwise co-assignment:** frequency with which two specimens share a cluster.
- **Exact-cluster recovery:** frequency with which a point-fit cluster's entire
  specimen set appears as one complete cluster in a replicate. Splitting,
  merging, or moving any specimen breaks exact recovery.
- **Selected K counts:** successful replicate counts for each selected K, with
  the number of failed replicates reported separately.

These are finite-replicate sensitivity measures for the declared locus-block
resampling distribution, **conditional on the sampled specimens**. They do not
resample biological individuals or quantify species truth. Unequal sampling and
continuous spatial structure can produce stable clusters. More replicates reduce
Monte Carlo noise but do not repair incorrect resampling or model assumptions.

All requested replicates are attempted exactly once. Failed fits retain their
seeds and errors; there are no silent retries or success-only support estimates.
If any replicate fails, support has `status: "unavailable"`, and no co-assignment
point matrix or scalar recovery frequency is published. For an observed count
`c`, `B` requested replicates and `F` failures, export bounds
`[c/B, (c+F)/B]` for the unresolved finite set of replicate outcomes.
Self-co-assignment is known to be 1. These are **failure bounds, not confidence
intervals**. With zero failures the bounds coincide with the bootstrap frequency.

Conditional GMM membership always remains separate from bootstrap stability.
Thresholding membership affects reported assignments, not the fitted partition
or stability statistic. P-values are explicitly `not_computed` because this
baseline has no justified species-boundary null. No confidence intervals are
claimed. A completed point fit can have unavailable bootstrap evidence; inspect
the stability status as well as the overall run status.

## Outputs and API

`summary.json` records assumptions, candidate BIC/convergence/sizes, selected K,
exclusions, missingness, bootstrap counts/failure bounds, interpretations, exact
configuration snapshots/hashes and library versions. Source hashes and staged
reader diagnostics are under `ingestion/`. `result.json` follows the core
assignment/evidence contract. `bootstrap_replicates.json` preserves seeds,
partitions, candidates and failures. No truth labels enter the provenance hash.

Arrays are pickle-free: `dosages.npy`, `labels.npy` (zero-based),
`membership.npy`, `pca_scores.npy`, `preprocessing.npz` (retained original indices,
means, scales, PCA transform and missingness), and `mixture.npz` (weights, means
and diagonal variances in reported cluster order). Pairwise arrays use summary
specimen order: `coassignment.npy` when available, and
`coassignment_lower.npy`/`coassignment_upper.npy` whenever bootstrap is requested.
No serialized Python estimator is required to reconstruct point memberships.
Failures after directory creation leave `failure.json`; `summary.json` is the
completion marker. CLI errors return 2; a completed analysis returns 0 even if
bootstrap evidence is unavailable.

```python
from pathlib import Path

from super_deli import load_config
from super_deli.io.config import read_json
from super_deli.models.config import BaselineConfig
from super_deli.workflows.genomic_baseline import parse_blocks, run_genomic_baseline

source = Path("examples/baseline/dataset.json").absolute()
summary = run_genomic_baseline(
    load_config(source),
    model_config=BaselineConfig.from_dict(read_json(source.with_name("model.json"))),
    block_ids=parse_blocks(read_json(source.with_name("blocks.json"))),
    base_dir=source.parent,
    output_dir="/tmp/new-baseline-api-run",
)
```

The numerical API `super_deli.models.pca_gmm.fit_pca_gmm` accepts a specimen-by-locus
0/1/2 dosage matrix with NaN missing calls, unique specimen IDs, a `BaselineConfig`,
and optional block IDs. Its returned `PointFit.mixture` retains original EM
component numbering; `component_order` maps it to canonical membership columns.

Fits run sequentially. Dense co-assignment uses quadratic memory in specimens;
the budget checks an approximate 32 bytes per pair for working arrays, excluding
genotype/PCA/GMM memory. Input copies and full SVD also limit genome-scale use.
No distributed, sparse, or out-of-core estimator is claimed. Scoped thread limits
are serialized within this module; use separate processes for concurrent work
that would otherwise change native thread settings or SNPio logging globally.

## Verification boundary

Tests cover panmixia, separated and overlapping generating populations, unequal
sampling, a geographic cline, linked/unequal blocks, missingness, degeneracy,
convergence failures, label invariance, seed reproducibility, failure denominators,
source preservation, supported real SNPio readers, and array reconstruction.
Even strongly separated simulated groups can subdivide in refits; tests check
the reported sensitivity rather than force perfect recovery. These small fixtures
are software/regression checks, not a calibrated power, type-I error or taxonomic
validation study. Independent empirical references and broader simulation
calibration remain necessary before species-delimitation claims or formal tests.
