# Predictive screening of grouped AIC mixtures

Super Deli provides an opt-in predictive screen for complete continuous specimen
morphology and one ecological observation per population/locality. It uses the
existing group-mean morphology PCA model. The original full-family study below
remains unchanged; the separately validated [shared-covariance extension](shared-predictive.md)
adds an explicit tied model. Ordinary AIC/BIC,
defaults, existing fitting and historical p-values remain unchanged.

The motivation is the [preceding diagnosis](aic-gaussian-diagnostics-results.md):
AIC frequently selected several components under Gaussian controls, and those
selected fits had negative observed independent predictive gain over K=1.
This screen is a separately validated model-selection heuristic. It is not a
modified AIC formula, significance test or species-count estimator.

See the [completed validation and recovery tradeoffs](predictive-aic-results.md).

## Selection rule

1. Fit the ordinary full-data pipeline and retain all K candidates and parameters.
2. Divide target units into five deterministic shuffled folds. Keep a locality's
   morphology specimens and its ecology row together. Do not stratify using truth.
3. For each training fold, re-ingest raw inputs and refit group means,
   standardization, both view PCAs, view normalization, fusion, joint PCA and the
   complete K search. Apply the frozen training map to its validation localities.
4. For each K, calculate validation-unit log-density differences against that
   fold's K=1 fit in the identical score space. Pool equally over units, rather
   than giving unequal-sized folds equal weight.
5. A larger K passes only if eligible on the full data and every fold and its
   pooled mean predictive gain is strictly positive. Select the smallest AIC
   among passing full-data candidates and K=1; break ties toward smaller K.

An ordinary AIC K=1 therefore stays K=1. If a larger candidate is rejected,
another passing candidate or K=1 can be selected. The selected memberships and
parameters come from the already retained full-data candidate; no additional
fit occurs after selection.

Small-component/infeasible candidates fail screening with their reasons retained.
Numerical failures, nonconvergence, rank failures or missing validation predictions
withhold the screened result. They never silently become a valid K=1 selection.
All attempted searches and partial exports are retained without replacement.

The positive-gain threshold is prespecified at zero and is not tuned using the
benchmark. Every data-adaptive transformation belongs inside its training fold,
as described in the [Stanford CV notes](https://web.stanford.edu/class/stats202/notes/Resampling/Kfold-CV.html).
Overlapping training folds complicate uncertainty estimation
([Bengio and Grandvalet, 2004](https://jmlr.org/papers/v5/grandvalet04a.html));
this implementation supplies no fold-independence SE, one-SE rule or CV p-value.

## Use and retained outputs

Use Python 3.11 or 3.12 with the `clustering` extra:

```bash
super-deli cluster-integrative-predictive /absolute/dataset.json \
  --model-config /absolute/model.json \
  --screen-config /absolute/screen.json \
  --output-dir /absolute/new-output
```

The examples in `examples/predictive-aic/` show the model and screen configurations.
The workflow accepts existing declared continuous morphology/ecology features,
preserves their identities/units and explicit population/locality links, and
requires both views to be complete. This point command disallows stability draws;
a separate [shared-family locality-bootstrap workflow](predictive-stability.md)
refits the complete screen under its narrower scope. The full-data
candidate search keeps existing minimum-size and best-start screening behavior.

`full/run/` retains the unmodified ordinary AIC pipeline. `fold_000/` through
`fold_004/` retain training raw data, configurations, candidate parameters,
validation identities and predictions. `fold_assignments.json` records all
splits. `summary.json` explicitly separates ordinary AIC, descriptive BIC and
screened selection, eligibility/gain decisions, failure status and fit accounting.
`selected_assignments.csv`, `selected_memberships.npz` and `selected_model.npz`
contain the screened full-data candidate when available. The CLI exits nonzero
when screening is withheld. Original AIC outputs remain available separately.

Membership probabilities are conditional on the chosen fitted mixture. They do
not quantify uncertainty in the screen itself. Shared-family, fixed-fold locality
resampling stability is available through the separate workflow linked above.
Full-family screen stability and significance for either screened policy remain
unsupported: existing calibration APIs do not accept the screened output/schema.
No historical p-values are overwritten or reinterpreted.

## Frozen validation

The approved study uses 625 fresh independent observations:

- 300 raw Gaussian controls: 50 each from independent, correlated and
  zero-conditional laws at 64 and 256 localities.
- 100 independent four-dimensional Gaussian score controls: 50 per sample size,
  drawn from N(0,I/4). These change the score law as well as removing preprocessing.
- 150 complementary four-group alternatives: 25 per N=64/256 crossed with
  signal multipliers 1.5, 3 and 4.5.
- 75 stress datasets at N=64: 25 each under heavy tails, continuous clines and
  unequal-variance four-group structure. Stress departures are not species truth.

All raw data retain 1–4 morphology specimens/locality, one ecology row/locality
and four traits/view. The study fixes two PCs/view, four joint PCs, equal weights,
full covariance, K=1..4, minimum distinct size five, ten starts, 500 iterations,
tol=0.001, reg_covar=1e-6 and full-data model seed 71323. Split and fold-fit seeds
are fixed before generation. Truth is evaluated only after fitting.

Each observation uses one full-data and five fold searches. Together with 25
verified archived baseline replays, the main budget is **3,775 searches**.
There are no significance or stability bootstraps. A separate full-settings
storage preflight uses one observation per 17 study cells: **102 searches**.

Every fresh observation has 2,048 independently generated external evaluation
units, projected through its full-training map. External data never influence
screening, folds, thresholds or settings. Comparisons of candidates are within
the same fitted score space; absolute densities are not compared across maps.

The primary family has 28 rates: ordinary/screened K>1 in eight control cells
and ordinary/screened K=4 in six alternative cells. Report exact pointwise 95%
intervals and Bonferroni simultaneous 95% intervals, Monte Carlo SEs, denominators
and withheld-outcome bounds. Fourteen paired rate differences receive adjusted
exact discordance bounds. ARI, predictive gains, BIC and stress outcomes remain
separate descriptive results. No CV fold is counted as an independent dataset.

Independent numerical audits reconstruct generating laws, all identities/counts,
fold subsets, training PCA parameters, validation/external projections, candidate
densities/membership/criteria, eligibility, pooled gains, selections and exports.
A separate standard-library checker decodes saved NumPy validation vectors and
checks archive/member bytes, fold gains, criteria and summary totals. Neither
auditor refits models. The benchmark reports recovery losses as well as reduced
overpartitioning; no new default is promoted automatically.

## Storage and reproduction

The main/probe configurations are in `examples/predictive-aic/`. Benchmark CLI
commands are `benchmark-predictive-aic`, `analyze-predictive-aic`,
`preflight-predictive-aic` and `replay-predictive-aic`; use `--help` for paths.
The installed smoke checks all 17 cells and independently replays the public CLI:

```bash
python examples/predictive-aic/smoke.py --output-dir /absolute/new-smoke
```

Full studies require a matching measured storage preflight. It projects verified
lossless retention with 25% growth plus 128 MiB reserve and requires a 2 GiB
checkpoint artifact cap and at least 8 GiB free. Runtime guards preserve partial
evidence on failure. Use at most four single-thread workers. Detailed raw data,
fold models, predictions and private indexes stay local; only source, settings,
documentation, figures and small aggregates are published. Figures additionally
require Matplotlib. Previous checkouts and all historical studies are preserved.
