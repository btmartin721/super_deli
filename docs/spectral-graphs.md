# Spectral graphs on frozen representations

This opt-in diagnostic tests supplied-K partitions on the completed denoising
archive. It does not select the number of species or introduce a production
spectral model. Existing model defaults, configurations, p-values and empirical
gates remain unchanged. Python 3.11/3.12 and the existing clustering extra are
supported; reading saved neural coordinates does not require neural training.

The approved scientific source contains 60 archived observations: Gaussian,
four spherical groups and two moons, ten each at N=64/256. Clean/noisy trait
tables are paired. The analysis is retrospective, not fresh performance
confirmation. No new scientific inputs, neural training, resampling,
significance tests or empirical fitting are included, and nobody is contacted.

## Fixed comparison

Each observation has 17 coordinate spaces: standardized generating 2D (oracle),
standardized traits in both conditions, native saved PCA in both conditions,
and standardized ordinary/denoising neural coordinates for both conditions and
seeds 0/101/202. Source coordinates never enter neural training. No new scaling,
raw-neural arm or favorable-seed selection is added.

For every space, construct Euclidean k-nearest-other-unit graphs at k=5 and 15.
Self-neighbors are excluded; exact distance ties follow source-ID order. The
affinity is (B+B.T)/2 for directed binary neighbor matrix B, with zero diagonal.
Request both K=2 and K=4 for every graph, regardless of generating labels.

Connected, numerically identified graphs use normalized spectral embedding
(ARPACK, fixed seed 71323, tolerance zero, first eigenvector retained) and the
library QR label assignment. The actual embedding, sparse graph and assignment
trace are retained. Exactly K connected components give a marked direct
component partition. Other disconnected cases, a cutoff eigengap <=1e-8,
ambiguous QR margins, failed solves or a different resulting label count withhold
the requested partition and retain its evidence. No repair or replacement is
attempted. Singletons/small groups remain visible; HDBSCAN's minimum size is
not imposed on spectral output.

The diagnostic reuses 900 saved HDBSCAN point partitions and adds directly
matched raw-eight-trait HDBSCAN fits, with its unchanged settings. Spectral
graphs count other neighbors; HDBSCAN min_samples=5 includes self. The comparison
changes graph construction, partition objective and noise policy together.

## Run the gated protocol

From a checkout of this version, install its optional numerical dependencies:

```bash
python -m pip install '.[clustering]'
super-deli diagnose-spectral-graphs --help
```

The diagnostic has `bind`, `freeze`, `preflight`, `audit-preflight`, `gate`,
`main`, `audit-main` and `report` actions. Each takes `--archive-root` and a new
`--output-dir`; `freeze` needs an approved plan at `approved-plan.md` in that
output root. `gate` requires measured `--staging-bytes`; execution allows
`--workers 1` or `--workers 2`. Optional `--config` accepts the separate schema-1
resource configuration. Resource limits cannot exceed the approved scope.

```json
{
  "schema_version": 1,
  "fixture": false,
  "max_workers": 2,
  "artifact_cap_bytes": 2147483648,
  "min_free_bytes": 8589934592,
  "max_projected_seconds": 3600,
  "max_working_bytes": 268435456
}
```

Source binding verifies inventories, identities, source bundle and protected
numerical modules, transforms and saved forward passes without relaxing the old
execution gate. Scientific replay uses the archived numerical runtime. A separate
`fixture=true` mode is explicitly invented and untrained; it cannot be frozen as
a scientific source. Its generated projections are not fitted neural networks.

The main budget is 4,488 graph/K evaluations and 140 density fits/replays:
4,628 planned evaluations. Some requests produce component partitions or remain
withheld, so evaluation totals are not eigensolver-fit counts. A separate
504-evaluation preflight must pass independent audits and project under 2 GiB
new retained evidence, one hour execution and at least 8 GiB free, using at most
two one-thread CPU workers. Timing conservatively allows every requested K to
require a solver; missing eligible timing strata close the gate. Storage includes
staging, verified copies, 25% allowance and 128 MiB reserve. Failed gates require
an explicit amendment. Prior archives remain in place.

## Outputs and interpretation

The immutable binding/source freeze, materialized coordinates, per-graph sparse
arrays, spectra, embeddings, QR traces, failures, density baselines, execution
ledgers and independent audits remain local. `aggregate.json` has requested and
available counts, withholding reasons, available-only ARI, all-request bounds and
paired comparisons on explicitly counted common-available datasets. Read these
with HDBSCAN noise/coverage and small-cluster counts. Full spectral assignment is
not evidence that every specimen was recovered meaningfully.

The all-assigned K=1 reference is a fit-free constant partition. Gaussian
partitions forced to K=2/4 are not false-positive tests; selecting two moons when
two groups were supplied is not discovery of K. No p-value, new resampling
stability, winner, default promotion or species-boundary claim is provided.
Generating labels are used only for post-fit evaluation. Paired tables, graphs,
seeds and replays do not enlarge the independent sample count. Favorable results
require separate fresh validation and an unknown-K strategy.

Audits independently check distance arithmetic, declared tie ordering, neighbor
graphs/degrees/components, dense eigenspectra, actual embedding residuals,
orthogonality/projectors, QR/SVD equations, margins, labels, exact replay and
complete accounting. They do not reimplement the entire library eigensolver or
HDBSCAN. Only code/docs, invented fixtures, figures and small aggregates are
public. See [scikit-learn's spectral description](https://scikit-learn.org/1.6/modules/clustering.html#spectral-clustering)
and [Damle et al. (2019)](https://doi.org/10.1093/imaiai/iay008).
