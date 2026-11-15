# Bounded matched multimodal validation

All 24 main observations and the separate two-observation preflight completed
without failed point analyses. Across 240 main and 20 preflight point pipelines,
3,192 and 266 numerical-array comparisons respectively matched standalone native
replays exactly. Independent densities and parameter counts checked 130 selected
mixtures/BIC values, with eight additional geometric-expression density audits.

The first main observation per law used three native stability draws per arm:
90 main bootstrap refits. Preflight added 30. Independent read-only observers
audited 100 observed point/draw shape fits through proper-rotation SVD alignment,
unit-centroid-size consensus, tangent projection and PCA orthonormality. Retained
raw-fit hashes matched every expected shape point/draw input, including specimen
multiplicities across paired views. All 120 draw records, seeds, inclusion/OOB
sets and coassignment denominators passed independent checks.

Eight separately labelled geometric point checks reused the complementary
preflight observation. Individual translation, proper rotation, positive scaling
and their combination preserved shape-only and combined PCA distance geometry
within frozen tolerances (relative 1e-7, absolute 1e-9) and preserved pairwise
cluster assignments. These transformed observations are not independent evidence
about recovery rates. Unit tests separately covered 2D TPS/CSV, 3D CSV, cohorts
without genomics, missing/duplicate observations, model-policy mismatches,
source/observation tampering, failed arms and unchanged legacy workflows.

Each fresh observation contained 64 independent invented specimens, 64 biallelic
SNPs, four continuous traits in each of two tabular views, and six 2D landmarks.
Six laws had four observations each: one continuous population, redundant,
complementary and conflicting signals, genomic-only signal, and shape-only signal.
Shape used a six-point reference configuration with small Gaussian coordinate
variation and, where specified, two fixed deformation directions orthogonal to
translation, rotation and scale. Genomic/tabular inputs reused the documented
invented generator. Source orders were independently shuffled. Settings/seeds
were frozen before generation; labels entered evaluation only after fitting.

All arms selected K=1 in the four continuous controls. Combined analyses selected
K=4 in all four complementary observations, where separate genomic and shape
arms each selected K=2. However, the combined analysis selected K=1 in all four
shape-only observations while shape alone selected K=4; it retained K=2 in only
two of four genomic-only observations. Equal view weights and two joint PCs can
lose signal when informative views are combined with uninformative views.
No configuration or model default was changed in response to these outcomes.

These are intentionally small engineering/operating examples, not calibrated
false-positive rates, broad power estimates or species validation. Independent
new seeds and an added view prevent a paired claim of improvement over the prior
checkpoint. Conflicting views have no privileged biological partition; arbitrary
latent labels in continuous controls are not four-group truth. Three stability
draws test integration only and provide no precise biological support estimate.

Complete scientific inputs, parameters, refit audits, replays and failure records
remain local. Only [small cell aggregates](../examples/multimodal/results/validation.json)
are published. Existing models/defaults, historical p-values, old schemas and
empirical evidence gates are preserved; no empirical fitting or author contact.
