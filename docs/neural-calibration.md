# Full-pipeline calibration for separate encoders and a shared GMM

```bash
python -m pip install '.[neural]'
super-deli calibrate-matched-separate-gmm /path/to/dataset.json \
  --calibration-config examples/neural-calibration/calibration.json \
  --output-dir /path/to/new-output
```

This opt-in command supports complete continuous morphology and individually
measured ecology from the same independent specimens, linked by explicit IDs.
Its configuration declares the raw-Gaussian assumption and one prespecified
separate-encoder/shared-GMM model. Existing point workflows, defaults,
serialization and historical p-values remain unchanged. Python 3.11/3.12 and the
optional neural dependencies are required.

The question is whether the observed clustering statistic is unusual under a
declared single-Gaussian model for the selected raw measurements. This is not a
test that the selected components are species, or that the selected K is correct.

## Raw null and complete refits

Before any neural preprocessing, fit the joint raw mean and full covariance by
maximum likelihood, using divisor N. Cross-view covariance is retained. N must
exceed the total number of selected traits and allow at least two minimum-size
components. Constant features and singular or numerically rank-deficient raw
covariances are rejected. Rank is checked on the correlation matrix with default
tolerance 1e-12; no covariance floor, ridge or shrinkage is added to the raw null.

Generate 99 independent Gaussian raw datasets using those fitted parameters held
fixed within the test. Preserve observation IDs, specimen links, feature names,
units and the original typed table row orders. Every draw repeats CSV ingestion,
feature handling, standardization, both 500-update native encoders, coordinate
standardization/fusion and the complete shared-covariance GMM search. Stored
observed network weights are never reused for null fits.

Neural and mixture seeds are separate, fixed declarations. A procedure that
chooses a seed, representation, covariance family or hyperparameters after
examining results is outside this calibration. The command accepts the existing
shared-GMM model settings, with subsampling disabled. Independent units are an
assumption supplied by the user, not verified by matching IDs. Grouped/related or
spatial sampling, missing values/views, count/categorical traits, genomics and
shape require different null models and are excluded here.

## Statistic, uncertainty and failures

The global statistic is `BIC(K=1) - min eligible BIC(K=1..Kmax)`. K=1 is included,
so the statistic is nonnegative and often zero. The existing component-size,
ten-start, convergence and candidate-failure policies remain unchanged.

For B requested draws and b null statistics at least as large as the observed
statistic, use `p=(b+1)/(B+1)`, including ties. B=99 gives resolution 0.01; the
default decision threshold is 0.05. Small invented installation examples use
fewer draws and explicitly report whether their resolution reaches alpha.

Because the raw-null parameters are estimated, p-values are approximate. The
reported exact binomial interval concerns Monte Carlo tail uncertainty conditional
on those estimates; it excludes estimation error and misspecification. Heavy
tails, gradients, outliers and other non-Gaussian structure can produce departure
without separate species. Non-rejection does not establish one species. There is
no general finite-sample size guarantee or cluster-specific significance.

A failed observed search leaves null draws unattempted. A failed requested draw
is retained, never replaced; the primary p-value and decision are withheld.
Failure counts and unknown-outcome bounds remain explicit. Algorithmic failures
are never converted to K=1. Sampling stability and conditional mixture memberships
remain distinct evidence types and are not recalibrated by this command.

## Artifacts and API

```python
from super_deli.io.config import load_config, read_json
from super_deli.models.neural_calibration_config import NeuralCalibrationConfig
from super_deli.workflows.neural_calibration import run_neural_calibration

dataset = load_config("/path/to/dataset.json")
policy = NeuralCalibrationConfig.from_dict(read_json("/path/to/calibration.json"))
result = run_neural_calibration(
    dataset,
    config=policy,
    base_dir="/path/to",
    output_dir="/path/to/new-output",
)
```

Outputs include fitted raw values/parameters and their layout, observed/draw
statistics, seeds, failure accounting, evidence scope and a lossless pipeline
archive with an integrity inventory. Compact numerical fits preserve the exact
native point arrays while avoiding thousands of redundant HTML plots. Run the
existing point workflow separately for its interactive report and frozen-model
loader. Existing output directories and observed inputs are never replaced.

The [bounded validation protocol](neural-calibration-study.md) compares the
neural workflow with a benchmark-only per-view PCA/shared-GMM reference using
paired raw draws. This does not promote PCA calibration to a new production
adapter or select a winning model.
