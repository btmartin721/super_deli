# Unified matched discovery: completed integration validation

All 18 main synthetic datasets and six separate preflight datasets passed their
orchestration, numerical and standalone-parity checks. Both native workflows ran
on the same complete specimen cohort, with ecology rows independently permuted
in source files. No new model, default, historical p-value or empirical gate changed.

The main phase performed 192 native pipeline fits, including point fits, bounded
native resampling and exact standalone replays. The preflight performed 48.
Across both phases, 240 branch trainings and **120,000 actual optimizer updates**
were independently checked. Standalone replays are engineering comparisons, not
additional independent datasets. All requested fits completed without scientific
failures or replacement datasets.

## Purpose and fixed design

This is a small software-integration validation, not a new power or model-ranking
study. The existing generator supplies independent Gaussian controls,
complementary four-group signals and nonlinear noisy moons. Each main law/N cell
has three independent datasets at N=64/128. One fresh dataset per cell formed the
separate preflight. Raw values, generating labels and independent reconstruction
equations are retained locally; labels never entered model fitting or selection.

The first main dataset per cell requested five native resamples per arm; others
were point-only. Preflight datasets requested one per arm. PCA uses its existing
whole-individual bootstrap, including multiplicities and OOB counts. The neural
workflow uses its existing paired 80% subsampling without replacement. Their
supports are not pooled or treated as species probabilities.

Both models retained two coordinates per view, four clustering coordinates,
shared covariance, K=1..4, minimum component size five and ten EM starts. Neural
initialization was fixed at seed zero; neither seeds nor models were selected on
validation outcomes. Configuration/runtime/native-library hashes and every
data/order/noise/resampling seed were registered before generation.

## Observed partitions

| Law | N | Datasets | PCA K counts | Neural K counts | Between-model ARI range |
|---|---:|---:|---|---|---|
| gaussian_independent | 64 | 3 | {'1': 3} | {'1': 3} | 1.000–1.000 |
| gaussian_independent | 128 | 3 | {'1': 3} | {'1': 3} | 1.000–1.000 |
| four_complementary | 64 | 3 | {'4': 3} | {'4': 3} | 0.918–1.000 |
| four_complementary | 128 | 3 | {'4': 3} | {'4': 3} | 1.000–1.000 |
| moons | 64 | 3 | {'2': 3} | {'2': 2, '1': 1} | 0.000–1.000 |
| moons | 128 | 3 | {'2': 2, '3': 1} | {'4': 1, '2': 1, '3': 1} | 0.590–1.000 |

K-count dictionaries count the three main datasets in each cell. ARI compares
the two fitted partitions, not their biological accuracy. ARI=1 for two K=1
partitions is a degenerate agreement convention, not evidence of recovered
species. These small cells give no general false-positive, recovery or neural
advantage guarantee. A selected K=4 reaches the configured search boundary.

## Independent checks and retained evidence

Checks reconstructed source measurement laws, ID joins, raw PCA standardization,
profile weighting and joint projections. Dense Gaussian calculations verified
selected-mixture memberships and BIC. Independent NumPy gradient/Adam references
checked every actual neural optimizer update and linked branch inputs, weights,
losses and draw identities. Every native numeric artifact and resampling record
was compared with a standalone replay. An external fit-free export audit checked
the resulting tables, partition comparisons, native evidence and artifact hashes.

Main execution plus its in-process independent audits took 215.04
serial seconds; the separate preflight took 57.79. The measured
preflight gate passed the sixty-minute main/audit ceiling and four-GiB local
evidence budget with an eight-GiB free reserve. Engineering tests, packaging,
publication/CI and the final evidence-copy audit are recorded separately.

The offline HTML report was visually checked using the invented 32-specimen
example. It identifies coordinate spaces, local component colors, search limits,
native resampling and uncomputed significance. Full raw data, fitted models,
optimizer traces, source freezes and CI logs stay local; only this report and
small aggregates are public. No empirical analysis or author contact occurred.

[CLI/API and evidence meanings](matched-analysis.md),
[small aggregate](../examples/matched-analysis/results/validation.json),
[reproducible validator](../examples/matched-analysis/validate.py).
