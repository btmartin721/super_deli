# Integrative benchmark smoke snapshot

This is a bounded validation run of `examples/benchmarking/smoke.json`: eight
scenarios, three independent datasets per scenario, three view-subset arms,
64 generating targets and ten paired bootstrap draws per fit. All 72 point
fits and 720 inner refits completed; no numerical failures were recorded.

The larger study configuration has not been run. These results do not establish
empirical species accuracy or support calibration. Zero estimated MCSE from
three identical outcomes is not evidence of zero simulation uncertainty. The
worst-case Bernoulli rate MCSE with three datasets is about 0.289.

| Scenario | Arm | K (dataset count) | Mean ARI | ARI MCSE | Coassignment error | OOB coverage | Wrong / stable clusters |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| null_matched | full | 1 (3) | 1.000 | 0.000 | 0.0311 | 0.760 | 0 / 0 |
| null_matched | genome_environment | 1 (3) | 1.000 | 0.000 | 0.0296 | 0.760 | 0 / 0 |
| null_matched | traits_shape | 1 (3) | 1.000 | 0.000 | 0.0436 | 0.760 | 0 / 0 |
| strong_matched | full | 4 (3) | 1.000 | 0.000 | 0.0000 | 0.760 | 0 / 12 |
| strong_matched | genome_environment | 4 (3) | 1.000 | 0.000 | 0.0001 | 0.760 | 0 / 12 |
| strong_matched | traits_shape | 4 (3) | 1.000 | 0.000 | 0.0000 | 0.760 | 0 / 12 |
| weak_matched | full | 1 (3) | 0.000 | 0.000 | 0.3298 | 0.744 | 0 / 0 |
| weak_matched | genome_environment | 1 (3) | 0.000 | 0.000 | 0.5773 | 0.744 | 0 / 0 |
| weak_matched | traits_shape | 1 (3) | 0.000 | 0.000 | 0.6116 | 0.744 | 1 / 1 |
| complementary_matched | full | 4 (3) | 1.000 | 0.000 | 0.0001 | 0.732 | 0 / 12 |
| complementary_matched | genome_environment | 2 (3) | 0.488 | 0.000 | 0.2423 | 0.732 | 5 / 5 |
| complementary_matched | traits_shape | 2 (3) | 0.488 | 0.000 | 0.2385 | 0.732 | 4 / 4 |
| conflicting_matched | full | 4 (3) | 1.000 | 0.000 | 0.0000 | 0.744 | 0 / 12 |
| conflicting_matched | genome_environment | 1 (3) | 0.000 | 0.000 | 0.2331 | 0.744 | 0 / 0 |
| conflicting_matched | traits_shape | 4 (3) | 1.000 | 0.000 | 0.0000 | 0.744 | 0 / 12 |
| missing_cells | full | 4 (3) | 1.000 | 0.000 | 0.0000 | 0.734 | 0 / 12 |
| missing_cells | genome_environment | 4 (3) | 1.000 | 0.000 | 0.0000 | 0.734 | 0 / 12 |
| missing_cells | traits_shape | 4 (3) | 1.000 | 0.000 | 0.0000 | 0.734 | 0 / 12 |
| missing_views | full | 4 (3) | 1.000 | 0.000 | 0.0146 | 0.750 | 0 / 5 |
| missing_views | genome_environment | 4 (3) | 1.000 | 0.000 | 0.0148 | 0.750 | 0 / 2 |
| missing_views | traits_shape | 4 (3) | 1.000 | 0.000 | 0.0149 | 0.750 | 0 / 5 |
| unequal_grouped | full | 4 (3) | 1.000 | 0.000 | 0.0000 | 0.772 | 0 / 12 |
| unequal_grouped | genome_environment | 2 (3) | 0.488 | 0.000 | 0.2413 | 0.772 | 5 / 5 |
| unequal_grouped | traits_shape | 2 (3) | 0.488 | 0.000 | 0.2341 | 0.772 | 4 / 4 |

Coassignment error is the mean squared difference from generating same-group
indicators over unordered off-diagonal pairs. OOB coverage is the fraction
of those pairs with a finite, unaffected OOB estimate. Wrong/stable counts
pool the three datasets for display only and have no independent-cluster
confidence interval. Stable means exact-set recovery frequency at least 0.9.

Full integration recovered the four generating groups in these complementary
datasets; each two-view arm recovered its two signal groups. Weak separation
produced K=1 in every point fit, and bootstrap coassignment error remained
substantial. The missing-view cohort retained 43, 49 and 44 of 64 targets.
All arms used those same cohorts. OOB pair estimates remained incomplete with
ten draws. High recovery on the constructed strong scenarios does not validate
biological boundaries, and these particular conflict scenarios favored the
three primary-signal views over ecology's alternative grouping.

The hand-computed tests additionally include a stable but wrong partition.
Such a case is a metric-validation check, not an additional simulated result.

See [the protocol and metric definitions](integrative-benchmarking.md) and the
[machine-readable snapshot](../examples/benchmarking/smoke-results.json).

Configuration SHA-256: `4fde340f69c8188ee26ec698b61f9f7fc5f668dd80e91ed568c88040d95c4aef`.

Full run-ledger SHA-256: `f6ea26068b19776fbf030bd30df759aaf689c8ededfe1b479e56c8ee91540a00`.

The local full ledger, fitted artifacts and generating files were retained
outside Git. The snapshot records their hashes, selected summaries and package
versions; it is not a deposited raw benchmark-data release.

Software: numpy 2.2.6, python 3.12.10, scikit-learn 1.6.1, scipy 1.15.3, snpio 1.7.7, super-deli 0.12.0.dev0.
