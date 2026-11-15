# Matched genomic–phenotypic orchestration validation

All 24 frozen main observations and the separate two-observation preflight
completed with no failed point analyses. Every arm exactly matched its native
standalone point replay: 240 main and 20 preflight point pipelines, with 1,416
and 118 exact array comparisons respectively. Independent Gaussian-density and
parameter-count calculations checked all 130 selected mixtures and their BIC;
partition agreement was independently checked against scikit-learn.

The first observation in each main law used three native stability draws per
arm: 90 main bootstrap refits. Preflight added 30. All draw counts, seeded
sampling identities, inclusion/OOB accounting, paired view source links and
failure denominators were audited separately. Three draws only test engineering
integration; they are not precise estimates of biological stability.

Each observation had 64 invented specimens, 64 biallelic SNPs and four continuous
traits in each phenotype view. Source rows were independently shuffled to check
explicit identity matching. Six laws had four observations each: a common
population; redundant signals; complementary signals; deliberately conflicting
signals; genomic signal plus uninformative phenotypes; and phenotype signal plus
uninformative genomics. Fixed seeds/settings were recorded before generation.
No molecular/empirical labels entered fitting, and no p-values were calculated.

The combined analysis selected K>1 in one of four common-population controls;
all singleton analyses selected K=1 there. Combined K=4 occurred in only one of
four complementary observations. Genomic-only and combined analyses both
selected K=2 in all four genomic-signal observations. Additional views therefore
do not guarantee improved recovery, and the coordinator does not fix the known
PCA/model-selection limitations. Four observations per law support integration
checks and illustrative outcomes, not general false-positive or power claims.

Generating-label agreement is retained locally only as an evaluation diagnostic.
The arbitrary four-valued label in common-population controls is not truth for
four biological groups; conflicting views do not define a single privileged
partition. No configuration was changed after outcomes or promoted as a default.
The raw generator is an engineering simulation, not an empirical species model.

Detailed inputs, fitted parameters, bootstrap records, replays and audit evidence
remain local. Only [small cell aggregates](../examples/matched-genomic/results/validation.json)
are public. Native study version metadata records 0.79.0.dev0 because editable
distribution metadata preceded the additive version bump; the frozen source
hashes identify the actual 0.80.0.dev0 source. Installed-package and CI checks
use the rebuilt 0.80.0.dev0 distribution. No scientific arrays were rewritten.

The checkpoint adds a coherent user-facing comparison, not a new estimator or a
species-boundary test. The genomic autoencoder remains experimental under the
preceding no-go decision. Future genomic–phenotypic neural work requires its own
bounded design and justification rather than automatic promotion.
