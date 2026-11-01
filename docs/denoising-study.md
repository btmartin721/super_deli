# Frozen denoising comparison

The approved checkpoint tests one fixed denoising objective. It does not tune a
model to the previous curved-moons findings. The experiment contains 60 fresh
independent source observations: ten each for Gaussian, four spherical groups
and two moons, at N=64/256. Existing two-coordinate generating laws and feature
maps are unchanged. For each source, retain two paired eight-feature tables:
the analytical map and the map plus Gaussian raw-feature measurement noise
(SD 0.15). The moons source itself remains noisy (coordinate SD 0.08).

Compare PCA/HDBSCAN, ordinary autoencoder/HDBSCAN and denoising
autoencoder/HDBSCAN, retaining initialization seeds 0, 101 and 202 for each neural
family. Training corruption SD 0.15 is in standardized-feature units and is
distinct from simulated raw-feature measurement noise. The standardized
two-coordinate source receives an oracle-reference density fit; neither its
coordinates nor generating labels enter neural training. The mapping contrast
changes dimension, metric and scaling; only the within-map noise pair isolates
the added measurement noise.

The first observation in each law/N cell receives 20 paired 80% whole-unit
subsamples per condition/arm. Together with fixed exact replays and legacy
checks, the budget is 2,618 main workflow fits (2,190 neural trainings), following
a separate 266-fit preflight (222 neural trainings). No scientific comparisons
are changed after seeing fresh outcomes. Source roots 93001, 93019, 93047 and
93053 separate source data, measurement noise, subsampling and training corruption;
namespaces 11/13 distinguish preflight/main. All derived seeds are retained.
Each denoising initialization has a registered corruption stream, shared across
its paired feature-noise conditions and held fixed within subsampling. Comparing
different denoising seeds therefore varies initialization and corruption streams
together; it does not isolate initialization alone.

Before the main run, an independent audit must pass and the measured preflight
must project at most 2 GiB new local evidence and two hours main runtime,
including 25% allowances and a 128 MiB storage reserve, with at least 8 GiB free
and at most two one-thread CPU workers. Staging and verified copies are counted;
installed environments/caches are accounted separately. Failed gates require an
explicit amendment. Freeze complete source/runtime/settings and the approved
plan before generating scientific inputs. Bind the original neural archive and
replay numerical arrays without changing its old source/runtime gates.

The module CLI offers `freeze`, `preflight`, `audit-preflight`, `gate`, `main`,
`audit-main` and `report` phases:

```bash
python -m super_deli.benchmarking.denoising_study --help
```

`freeze` requires a new local evidence root containing `approved-plan.md` and
`--archive-dir` pointing to the completed original neural study. The gate uses
`--staging-bytes` for measured retained staging evidence. These are research
study commands; ordinary analysis uses `cluster-tabular-denoising`.

Report correct K, full-unit/assigned-only agreement, coverage, noise, splits,
merges, overpartitioning, reconstruction, source-neighborhood retention and
density connectivity separately. Seeds/tables/subsamples are repeated
measurements, not additional independent datasets. Native stability has only
one underlying observation/cell and is descriptive. Preserve failed attempts
without replacements, and give failure-aware bounds.

Seed zero is prespecified for 36 primary correct-K rates and 12 paired
denoising-minus-ordinary differences. Simultaneous intervals use 60 exact
binomial intervals with Bonferroni alpha 0.05/60: 36 rates and two discordant
categories per paired difference. Ten observations/cell yield broad uncertainty;
other seeds and metrics remain descriptive. Correct K does not establish group
recovery. Gaussian overpartitioning is not calibrated species false-positive
error. No new p-values, empirical fitting, author contact, automatic winning
seed/model or default promotion is included.

Audits independently regenerate source/noise/draw streams, verify identities,
all corruption digests, NumPy forward objectives and the first Adam update,
PCA parameters, density invariants/distances, numeric replays and inclusion
denominators. HDBSCAN is not independently reimplemented. Full inputs, weights,
noise/draw traces and verification logs stay local; only code, documentation,
invented fixtures, figures and small aggregates are public.
