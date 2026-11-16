# Bootstrap splitting and cluster-size eligibility diagnosis

This benchmark-only command examines ordinary 100-start, shared-covariance GMM
fits with full retention of eight selected view-PC coordinates. Production
models, defaults, serialization and historical p-values are unchanged.

```
super-deli diagnose-bootstrap-splitting configuration.json \
  --source /absolute/path/to/completed-truth-free-study \
  --generator /absolute/path/to/examples/multimodal/synthetic.py \
  --output-dir /absolute/path/to/new-local-study --phase cache
```

Run `preflight` with the same arguments next; `main` requires its passed resource
gate and unchanged source/runtime freeze. Configuration uses exact fields from
`BootstrapSplitConfig.to_dict()`, including both source-seal SHA256 checksums.
Invented seals cannot authorize the scientific study. Output directories are
exclusive; failed evidence is preserved rather than overwritten.

The frozen design includes 24 independently generated multimodal parents and 24
eight-dimensional standard-Gaussian score parents, at N=64/256. Raw parents contain
fixed-panel diploid genotypes, continuous morphology/ecology and fixed landmarks;
controls have one population, while alternatives have four known shape groups.
Alternatives carry shape signal only; the other three views are nonsignal. Their
generating group counts are balanced and fixed by design; known-parameter
reference cohorts preserve that design, while ordinary bootstrap changes realized
group representation. This is not an unknown-label resampling solution.
The raw law uses 64 independent Binomial(2,p) loci with parent-specific
p drawn uniformly from 0.2 to 0.8, four independent standard-normal traits per
tabular view, and six planar regular-hexagon landmarks with coordinate-noise
SD 0.025. The alternative adds the four sign combinations of two fixed
orthogonal shape modes at amplitude 0.18. These are invented geometry controls,
not anatomically realistic specimens or empirical species truth.
Source parameters stay fixed across an observed sample, 25 independent reference
cohorts and 1,024 independent evaluation units. VCF REF=A/ALT=C keeps genotype
coding consistent between separately ingested cohorts. Bootstrap draws reuse
native ingested arrays, as in the existing integrative workflow.

Each parent compares ordinary paired N-out-of-N bootstrap draws with their exact
unique original IDs and independent reference cohorts. Unique support changes
weights, sample size, BIC and preprocessing together; it does not isolate any one
of these mechanisms. Standard-Gaussian score controls bypass ingestion/PCA and
are explicitly head-only. The evaluation cohort is never used to choose settings
or components. Predictive density gains are relative to K=1 within each fitted
coordinate space, not comparisons of absolute densities across transformations.

The design requires 3,856 main and 660 separate preflight searches, plus a fit-free
trace of 2,250 archived draws. Main searches include eight exact whole-workflow
replays. Native minimum cluster size remains nine distinct original individuals;
one winning estimator per K supplies eligibility, as before. No alternative-start
salvage, covariance change, new p-value or new bootstrap method is introduced.

All training-candidate labels and selected external assignments receive strict
independent density/label checks. Numerical disagreement withholds that diagnostic
search and remains in failure bounds; it never becomes K=1. Structural or source
identity failures stop execution. No replacement draws or post-outcome tuning.

Detailed genotypes, traits, coordinates, labels, candidate parameters, predictions
and native bootstrap evidence remain local. Public results contain small parent-
level aggregates. Draws are conditional repeated measurements; uncertainty uses
independent parents and the small per-cell samples limit generalization. Stability,
component counts and model departure are not probabilities of species boundaries.
No empirical fitting, new significance tests, neural training or author contact.
