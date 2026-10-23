# Grouped four-group recovery diagnostics

This checkpoint diagnoses shared-covariance integrative clustering of continuous
specimen morphology and group-level ecology. It changes no production model,
default, calibration p-value, or existing artifact. It neither selects a recommended
configuration automatically nor runs stability/significance bootstraps.

## Frozen design

The source is the 25 complementary observations from the shared grouped calibration
study. SHA-256 pins identify its declaration, full file inventory and runtime source
manifest. Scientific source compatibility is checked before staging; every selected
source file (including its 99-draw archive) is checksummed. Observations, original
identities and labels are independently regenerated and the archived point outputs
are audited. No original calibration is rerun.

Before fresh generation, the runner writes its configuration, source pins and full
grid. Master seed **20261104** gives 25 fresh observations with 64 groups and 25 with
256 groups. These are independent cohorts, not paired across sample sizes. The
unchanged complementary simulator supplies four equally sized generating groups,
independent non-informative morphology counts from 1 through 4, four traits/view,
and exactly one ecology row/group. Morphology carries axis a and ecology axis b,
on top of correlated Gaussian between-group and within-morphology variation.

Each observation receives these nine shared-covariance point analyses:

- Baseline: two PCs/view, two joint PCs, equal view weights, 10 EM starts.
- Joint dimensions 3 and 4, retaining two PCs/view.
- Ecology weights 0.5, 2 and 4, retaining all other baseline settings.
- 100 EM starts, retaining the baseline score space.
- Four PCs in each view, with joint dimensions 2 and 4.

A tenth point analysis uses baseline settings and the **first declared morphology
specimen per group**, keeping its original specimen/observation IDs and every
original ecology row. This is a paired sensitivity control: removing replication
changes measurement precision as well as unequal source weighting. It does not
identify the causal effect of weighting alone and does not add a new preprocessing
method. No truth or cluster assignment determines which specimen is retained.

All settings retain K=1..4, minimum hard cluster size 5, fixed model seed 71323,
10 starts except the declared 100-start arm, maximum 500 iterations, tolerance
0.001 and regularization 1e-6. There are **750 planned point workflows** and
**75 separate truth-initialized K=4 optimization diagnostics**. The latter initialize
means/weights from generating labels and the single covariance from pooled residuals;
EM then runs unrestricted within the shared family. They never enter unsupervised
selection or alter p-values. Failures are retained without replacement; a failed
baseline leaves its truth-initialized fit unattempted. Archived replay failures
are reported and prevent interpretation as a reproduced baseline.

## Diagnostic quantities

Trace axis predictability (ordinary descriptive linear R-squared) through raw group
means, source-level PCA scores, group profiles, fused profiles and joint PCA scores.
Repeated specimen labels in source-level summaries are descriptive, not independent
replicates for inference. Record source PCA loadings/variance, joint loadings by view,
counts by generating group, count/axis association, selected K, adjusted Rand
agreement, axis homogeneity, candidate convergence and hard-size eligibility.

For each fixed score space, decompose BIC into twice log likelihood and its parameter
penalty. A tied K-component, d-dimensional GMM has
`(K-1) + K*d + d*(d+1)/2` parameters. Compare K candidates **within the same fitted
score space only**. Absolute BICs across different dimensions, weights, preprocessing
or specimens are not comparable evidence for choosing a configuration.

## Run and audit

Install `super-deli[clustering]` under Python 3.11 or 3.12. SNPio is unnecessary for
these two continuous views. Use a new output directory outside the source archive:

```bash
super-deli diagnose-grouped-recovery examples/grouped-recovery/study.json \
  --source-artifacts /path/to/shared-grouped-null-artifacts \
  --output-dir /path/to/new-diagnostic-study --workers 4
super-deli analyze-grouped-recovery /path/to/new-diagnostic-study \
  --output-dir /path/to/new-diagnostic-audit
```

Workers are restricted to 1..4; fits use one numerical-library thread. Pin the
scientific wheel and dependency versions for exact replay. The included smoke
script creates a small current-runtime source archive, runs both installed CLIs and
checks 30 point workflows and three truth-initialized fits.

The separate audit verifies every file hash, regenerates raw data/identities and the
one-specimen inputs, checks counts, settings, seeds, complete ledgers and archived
replay, and uses independent raw moments/eigen-equations to validate source PCA.
It checks group aggregation/fusion/projection and selected GMM likelihood,
membership and BIC with dense Gaussian calculations. ARI uses an independent
contingency formula; axis recovery uses pseudoinverse projections. Truth-based
initial means, weights and pooled covariance are independently reconstructed.
No model is refitted by the audit. Nonselected GMM likelihoods are reconstructed
from their BIC logs; their parameters and full EM histories are not independently
available. Numerical convergence is not a certificate of global optimality.

Pointwise Wilson intervals describe K=4 selection frequencies under this one
constructed law. They are unadjusted for the diagnostic grid. K=4 does not imply
correct partitions; ARI and axis retention provide distinct checks. Existing global
model-departure p-values establish neither K nor species boundaries. Any proposed
model/default change or recalibration belongs to a separately approved checkpoint.
