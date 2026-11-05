# Frozen-coordinate diagnosis of matched neural integration

`diagnose-neural-coordinates` is a separate experimental diagnostic for the
completed separate-encoder archive. It reuses all 80 archived observations and
ten saved spaces. It performs zero neural training, new data generation,
resampling or significance testing; defaults, existing models and historical
p-values retain their behavior. Python 3.11/3.12 and the `clustering` extra are
supported. Neither PyTorch nor SNPio is imported by the diagnostic runner.

Use a new local output root containing the approved plan as `approved-plan.md`.
The lifecycle actions are `bind`, `freeze`, `preflight`, `audit-preflight`,
`gate`, `main`, `audit-main` and `report`. Every action requires `--archive-root`
and `--output-dir`; `gate` also requires measured `--staging-bytes`. Scientific
runs require the source's original NumPy, SciPy, scikit-learn and threadpoolctl
versions for exact source assignment/strength/hierarchy parity.

The main budget is 2,400 evaluations: 800 unchanged HDBSCAN replays and 1,600
existing GMM searches on exactly the same two/four coordinates. Full and shared
covariance are prespecified separate comparisons, with BIC, K=1..4, minimum
hard size five, ten starts, 500 iterations, tolerance 0.001, regularization 1e-6
and model seed 71323. Candidate failures withhold the incomplete search, retain
partial evidence and are never replaced or converted into K=1. The existing
winning-start/admissibility policy is preserved. No production neural/GMM
adapter or covariance-family selection is added.

The separate 480-evaluation preflight uses the first archived observation in
each law/sample-size cell. Main repeats those computations; the repeated phases
and seeds are not new independent datasets. Existing no-root EOM and leaf
counterfactuals are derived from retained density trees without graph refitting.
Generating labels enter evaluation only, including branch purity, neighborhood
mixing and K=4 candidate agreement. No labels initialize or select fitted models.
A poor single-distance-threshold separation ratio is not proof that information
is absent or that every clustering method must fail.

Independent audits verify raw-to-frozen transforms, identities, distance/core/MST
and hierarchy invariants, source density parity, counterfactual library selection,
Gaussian densities/membership, convergence, hard-size eligibility, parameter
counts and within-space BIC. Absolute BIC values are not compared across coordinate
spaces or dimensions. Every source and generated artifact is fingerprinted.

Report all 480 law/N/space/clusterer groups. Primary descriptive rates concern
separate encoders under six signal/control laws, two sample sizes, three seeds
and three clusterers: 108 rates and 72 paired GMM-versus-HDBSCAN contrasts.
Keep coverage, noise, failures, intervals and failure bounds. Four-group recovery
requires K=4 and full ARI >=0.8, treating noise as unresolved singletons. Gaussian
K>1 splitting remains descriptive and uncalibrated. Discordant view partitions,
moons and hierarchy counterfactuals remain separate. Five previously examined
datasets/cell support retrospective diagnosis, not fresh confirmation or a
species guarantee. The deliverable identifies a supported mechanism, competing
explanations or an unresolved diagnosis and at most one targeted follow-up.

Before main, an audited measured preflight must project at most 1 GiB of retained
evidence and less than one hour for serial main execution plus audit, with a 25%
allowance, a 64 MiB reserve and no assumed parallel speedup. Require 8 GiB free
and at most two one-thread workers. Count logs, test archives, generated scratch
and verification copies. Dependency environments are tooling, not research
evidence; their disk use still reduces the free-space reserve and is recorded
separately. Guard failures remain retained and stop dependent work.

Detailed inputs, coordinates, fitted candidate parameters, assignments and per-fit
audits remain local. Public exports contain code, docs, figures and small cell
aggregates; audit exports contain phase-level counts and hashes. Existing
empirical sampling/reference gates remain closed. No author contact is involved.
