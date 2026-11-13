# Observed-feature feasibility diagnostic

This separately versioned benchmark asks whether the twelve measured continuous
morphology/ecology traits can support the specified reference/unfamiliar
distinction under deliberately favourable training. It adds no deployable detector.
Existing classifiers, support policies, schemas, defaults and p-values are unchanged.

One fixed 100-tree binary gradient-boosting classifier receives measured columns
only. It sees 1,024 known plus 512 nearby and 512 distant specimens during training,
and an equally sized independent calibration cohort. This explicitly supplies
unfamiliar labels and extra specimens unavailable to a known-only application.
Latent coordinates, size, true parameters, IDs, role labels and original classifier
probabilities are excluded from its predictors. Numeric artifacts use no pickle.

The original logistic and residual heads at seeds 0/101/202 remain frozen. A
diagnostic score may accept/withhold their original argmax but cannot relabel it.
A fixed 101-threshold grid is evaluated on calibration data: retain the threshold
with greatest known coverage meeting the original joint target, breaking ties
toward the largest threshold. If none qualifies, retain
`available_no_operating_point`; there is no fallback or numerical failure. Test
curves remain descriptive and never select an operating point.

Fresh evaluation separates known, nearby, distant, feature-identical unfamiliar
and shifted-known specimens. Compare feature-identical unfamiliar acceptance
with the matching known reference's coverage. The method gives no guarantee
for arbitrary unseen populations, species status, or changed sampling conditions.
A negative diagnostic does not prove that every method must fail.

The command deliberately takes a completed source study rather than empirical
input tables. Each phase verifies its sealed source/seed/configuration/runtime
bindings and original model inventories. Scientific processes prohibit Torch
and SNPio imports. Native tree stages, original predictions, decisions, metrics
and uncertainty are independently audited. Failed guards and incomplete results
are retained; completed work is verified and skipped on continuation.

```bash
super-deli benchmark-observed-feasibility bind \
  --archive-root /path/to/completed-class-local-study --output-dir observed-study
super-deli benchmark-observed-feasibility freeze \
  --archive-root /path/to/completed-class-local-study --output-dir observed-study \
  --study-config examples/observed-feasibility/study.json
```

Before freeze, place the approved experiment plan at
`observed-study/approved-plan.md`. Then use `run` and `audit` with
`--phase preflight`, followed by `gate --staging-bytes MEASURED_BYTES`. Only a
passing measured gate admits `run`/`audit --phase main`, followed by `report`.
Use at most two single-thread workers. Detailed records, scores, trees and logs
belong in local study storage; publish only small aggregates.

The lifecycle API is `engine`/`gate` in
`super_deli.benchmarking.observed_feasibility_study`; configuration is
`ObservedFeasibilityStudy` in `observed_feasibility_config`. The separate benchmark
contract does not reinterpret historical artifacts. A complete invented installed
check, including invention of source neural heads, needs the neural extra:

```bash
python examples/observed-feasibility/smoke.py --output-dir invented-observed-check
```

Invented installation fixtures are never scientific evidence. The main study
reuses archived numeric heads and performs no neural/logistic retraining.
See the [protocol](observed-feature-feasibility-validation.md).
