# Observed-data Gaussian mixtures

```bash
super-deli cluster-tabular-observed examples/tabular-observed/dataset.json \
  --model-config examples/tabular-observed/model.json \
  --output-dir /absolute/path/to/new-observed-gmm-run
```

This opt-in point workflow accepts one incomplete continuous morphology or ecology
table, with 2–8 explicitly selected analytical features and one row per unique
biological unit. It uses the existing typed ingestion and identity contracts.
Counts, categories, repeated-unit aggregation, mixed modalities and PCA are outside
this workflow. Existing complete-data and missing-policy analysis commands remain
unchanged. Install the `clustering` extra and use Python 3.11 or 3.12.

## What is fitted

Each specimen contributes a Gaussian-mixture density for its observed coordinates.
The EM algorithm uses component-conditional expectations and **conditional
covariances**, rather than fitting a single mean-filled table. This follows the
[observed-data EM formulation](https://arxiv.org/html/1209.0521), section II.
Mean filling is used only to initialize component locations. Missing source
measurements remain missing in the retained raw arrays.

Training-only observed-value means and standard deviations define fixed numerical
coordinates. Each selected feature and each feature pair must be observed in at
least five rows. Unobserved or constant features fail explicitly; all-missing rows
are excluded with unresolved assignments. These guards do not prove mixture
identifiability. At least max(5, dimension+2) informative units are required.

Component covariance eigenvalues are constrained to at least `covariance_bound`
(default 0.0001) in these fixed coordinates. The M-step solves this constrained
problem; it does not add the legacy `reg_covar` diagonal term. The constraint
prevents covariance collapse but changes the model, depends on the feature parameterization and
training scaling, and may affect selection. Inspect candidate spectra and active
bounds. Gaussian-mixture fitting remains a local-optimization problem.

The separate version-1 `ObservedGMMConfig` requires `feature_names`. Defaults are
K=1..4, ten starts, 2,000 iterations, likelihood tolerance 1e-7 per unit, relative
EM fixed-point tolerance 1e-5 and a minimum of five hard-assigned units/component.
For each K the best converged likelihood is considered before the size rule;
lower-likelihood starts do not rescue an undersized component. All starts and
failures are retained. If a feasible K has no converged fit, selection is withheld.
A failure is never silently reported as K=1.

BIC chooses among eligible candidates in this fixed feature space, with the usual
full-mixture parameter count and the number of informative rows. It is a heuristic:
mixture nonregularity, missing information and active covariance constraints limit
ordinary asymptotic arguments. Do not compare its absolute value across different
cohorts, dimensions or observed-coordinate patterns.

## Assumptions and interpretation

The likelihood assumes independent units, Gaussian mixture components and
ignorable MCAR/MAR missingness with distinct missingness parameters. A source
label or unique ID does not establish independence. MAR cannot be established
from the incomplete table alone; value-dependent MNAR needs other assumptions.

Membership measures assignment uncertainty **conditional on fitted parameters**.
It is not a species probability, confidence interval or significance level. No
bootstrap, p-value, missingness-model uncertainty or parameter-uncertainty interval
is provided in this checkpoint. Empirical reference and sampling gates remain
closed. Simulation success does not validate empirical species delimitation.

## Local outputs and projection

A new output directory contains a byte-preserved source snapshot, typed ingestion,
full eligibility ledger, raw masks, all candidate/start parameters and likelihood
traces, membership CSV, checksums, JSON summary and a local HTML report. All-missing
rows and failed searches remain visibly unresolved. No filled source table is
exported. Source/snapshot changes during fitting invalidate the run.

```python
import numpy as np
from pathlib import Path
from super_deli.io.config import load_config
from super_deli.models.observed_config import ObservedGMMConfig
from super_deli.workflows.observed_gmm import run_observed_gmm, load_observed_model

source = Path("examples/tabular-observed/dataset.json").resolve()
dataset = load_config(source)
configuration = ObservedGMMConfig(feature_names=("trait_0", "trait_1", "trait_2", "trait_3"))
summary = run_observed_gmm(
    dataset, model_config=configuration, base_dir=source.parent,
    output_dir="/absolute/path/to/new-run", modality_name="traits",
)
model = load_observed_model("/absolute/path/to/new-run")
# Numeric columns must follow the training feature order; NaN denotes missing.
new_values = np.array([[0.0, np.nan, 1.0, 2.0]])
log_density, membership = model.predict(new_values)
```

Prediction uses frozen training transformations and observed-coordinate marginals.
All-missing predictions are unresolved NaNs. Densities include each row's scaling
Jacobian so they are expressed in original observed-feature units. Keep matching
observed coordinates when comparing predictions across models.

## Bounded validation

The [simulation results](tabular-observed-results.md) compare four raw-feature
arms using the same constrained solver and common masked-training transform:
observed-data EM, complete cases, training-mean filling and a simulation-only
complete-data reference. This deliberately avoids confounding missing-value
handling with removal of PCA or a different covariance regularizer. These arms
are not exact numerical replacements for legacy PCA/GMM workflows.

Sixty independent four-feature datasets cover spherical/correlated Gaussians and
four Gaussian groups at N=64/256, each with five paired random, observed-trait,
hidden-value or structured masks. Settings and seeds were frozen before generation.
There are 1,200 point searches and 24 exact replays, following a separate audited
144-search preflight. Independent external units assess prediction without tuning.
Detailed evidence stays local; only small aggregates and figures are packaged.
Ten datasets/cell support descriptive comparisons, not broad superiority or
calibration claims. Standalone point fitting is the endpoint; integration into
unified reports and resampling/calibration require later checkpoints.

### Reproduce the frozen benchmark

Run the public phases against a new local directory, using an installed matching
package with the clustering extra. The freeze phase writes settings, versions and
source hashes before any scientific data are generated:

```bash
python examples/tabular-observed/run.py freeze --output-dir /absolute/path/to/new-study
python examples/tabular-observed/run.py preflight --output-dir /absolute/path/to/new-study
python examples/tabular-observed/run.py audit-preflight --output-dir /absolute/path/to/new-study
python examples/tabular-observed/run.py gate --output-dir /absolute/path/to/new-study
python examples/tabular-observed/run.py main --output-dir /absolute/path/to/new-study
python examples/tabular-observed/run.py audit-main --output-dir /absolute/path/to/new-study
python examples/tabular-observed/run.py report --output-dir /absolute/path/to/new-study
python examples/tabular-observed/audit_report.py --phase main --output-dir /absolute/path/to/new-study
```

The main phase checks the measured gate and frozen source hashes. A failed gate
requires review; it does not change the budget. `--resume` reuses completed
searches only after exact input/configuration/identity checks and never replaces
a failed numerical search. Partially written searches require attention instead
of automatic refitting. Candidate/start arrays retain optimization component
order; selected model arrays and CSV memberships use first-assigned-source-row
order. Cluster numbers are arbitrary labels.
