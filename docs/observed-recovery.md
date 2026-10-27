# Observed-data GMM recovery and convergence diagnostics

This benchmarking workflow distinguishes optimizer exhaustion, initialization
sensitivity and BIC component-selection cost in the existing observed-data GMM.
It adds no model, empirical analysis, significance test or bootstrap. The
production observed-coordinate EM, default 2,000-update budget, ten starts,
information guards, covariance constraint and historical results are unchanged.

The workflow requires the completed local observed-data study. It checksums
that evidence and preserves all original results, including withheld searches.
Detailed raw coordinates, parameters, traces and predictions stay outside Git.
Only reproducible code, invented engineering fixtures, figures and small
aggregates are published. There is no author contact.

## Architecture and continuation

`benchmarking.observed_recovery` resumes saved exhausted paths through the
unchanged `models.observed_gmm.fit_start`. It validates the exact final-parameter
and objective anchor and retains the original trace prefix. Each eligible path
gets at most 8,000 additional updates, for 10,000 cumulative updates. Converged
starts and starts failing for another reason are preserved without alteration.
Tolerances, objective, covariance lower bound and size policy do not change.

Extended-budget candidates are reconstructed in a separate output tree. For each
K, choose the highest-likelihood converged start, then require at least five hard
assignments per component. A lower-likelihood start cannot rescue this rule.
An unresolved feasible K withholds the entire extended-budget choice. The
original withheld result never becomes a successful production result.

The archived runner explicitly restricts continuation to K=4; fresh diagnostics
continue every exhausted feasible K as prespecified. The three archived
iteration-limited K=4 searches contribute 30 continuation attempts. The separate complete-case input with four rows remains rejected by
the existing information guard; it is not an optimizer problem.

`observed_recovery_study` freezes source identities, software, settings and
archival checksums before fresh generation. `observed_recovery_audit` calls the
existing independent density/conditional-moment reference and reconstructs
fresh inputs separately from the generator. It checks trace anchoring, first
continued update, final updates/residuals, covariance constraints, membership,
candidate selection, total budgets, failures and exact replays. The auditor
does not call production EM or selection. `observed_recovery_report` evaluates
labels after fitting and separates original/extended and oracle evidence.

## Frozen confirmation design

Sixty fresh independent datasets comprise ten each from spherical Gaussian,
correlated Gaussian and four-group laws at N=64 and N=256, with four continuous
features. The laws are unchanged from the completed observed-data study.
Generating group counts are random, not forced equal.

Use paired 30% independent missingness and an observed-trait MAR mask. Under MAR,
trait one is always measured; the other traits have missing probability 0.15
when trait one is nonpositive and 0.65 otherwise. The missing-data arm and its
complete-data reference share the mask's training-only observed-value scaler.
Hidden entries never determine the missing-data arm's transformation. No PCA or
completed-value substitution enters observed-data fitting.

There are 240 fresh baseline searches and 12 archived exact replays. Only
iteration-limited fresh starts are continued, capped at 9,600 attempts; actual
counts can be smaller. Each dataset also has 2,048 independent external units
with new masks for frozen prediction. External outcomes do not select or tune
models. Data/mask/evaluation seeds are 81403/81409/81429, with separate preflight
and main namespaces; the unchanged optimizer seed is 80423.

The 20 archived and 20 fresh four-group datasets have 160 separate K=4 diagnostic
starts across both mask/arm pairs. Initial weights, means and covariance come
from the known generating law in the same training-scaled coordinates. These
fits use the unchanged EM and a 10,000-update cap. They never enter unsupervised
selection or another fit's initialization. They are neither guaranteed global
optima nor an available empirical solution.

## Reproduction and storage gate

Install the packaged scientific extras under Python 3.11 or 3.12. Run the example
or its installed module, always with a new output directory outside the checkout:

```sh
python -m super_deli.benchmarking.observed_recovery_study freeze \
  --output-dir /absolute/local/new-study \
  --archive /absolute/local/completed-observed-study
python -m super_deli.benchmarking.observed_recovery_study archive-audit \
  --output-dir /absolute/local/new-study
python -m super_deli.benchmarking.observed_recovery_study preflight \
  --output-dir /absolute/local/new-study
python -m super_deli.benchmarking.observed_recovery_study audit-preflight \
  --output-dir /absolute/local/new-study
python -m super_deli.benchmarking.observed_recovery_study gate \
  --output-dir /absolute/local/new-study
python -m super_deli.benchmarking.observed_recovery_study main \
  --output-dir /absolute/local/new-study
python -m super_deli.benchmarking.observed_recovery_study audit-main \
  --output-dir /absolute/local/new-study
python -m super_deli.benchmarking.observed_recovery_study report \
  --output-dir /absolute/local/new-study
```

The separate full-settings preflight contains 24 baseline searches, six exact
replays, at most 960 continuations and eight known-parameter starts. Main fitting
requires its independent audit and a passed measured storage gate. Projection
includes a conservative maximum extra-trace/metadata budget, 25% allowance and
128 MiB reserve. The total new evidence cap is 2 GiB, with at least 8 GiB free
and at most four single-thread workers. An original archived-scope filter error and its failed audit are retained in
the completed study. The scope repair reused approved K=4 trajectories without
refitting and excluded 24 extra paths. An immutable execution-freeze record
binds the repaired orchestration; numerical updates and generators did not change.
The runner checks space before and after
each observation and before archived jobs. No automatic partial-run resume or
replacement fit is implemented: an existing phase directory blocks rerunning it.
Retain partial evidence and resolve an interruption explicitly.

## Interpretation

Compare likelihood gains and BIC penalties only within the same fitted
coordinates, cohort and mask. Report candidate versus selected ARI, component
eligibility, bounds, covariance conditioning, numerical failures and external
prediction separately. Better truth agreement does not identify a deployable
model. A converged EM solution is not certified globally optimal.

Datasets are the independent units. Masks, references, starts and budgets are
repeated measurements. Ten datasets per cell give wide pointwise intervals;
these are not simultaneous guarantees. Unresolved searches remain in requested
counts with explicit outcome bounds. Finite-sample BIC cost, slow convergence,
local optima and information loss can coexist.

A shared-covariance observed-data model or a production optimizer/default change
requires a later checkpoint. Empirical sampling/reference gates remain closed.
See the [audited findings](observed-recovery-results.md).
