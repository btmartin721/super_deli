# SNPio-backed genomic neural baseline

This separate, opt-in workflow discovers candidate genetic clusters in complete
diploid biallelic calls. It does not delimit biological species by itself. It
leaves the original genomic PCA baseline and all significance workflows intact.

## Run the invented example

Use Python 3.11 or 3.12 in an isolated environment, from the repository root:

```bash
python -m pip install -e '.[genomics,neural]'
super-deli analyze-genomic-models examples/genomic-neural/dataset.json \
  --model-config examples/genomic-neural/analysis.json \
  --output-dir genomic-comparison-01
```

Open `genomic-comparison-01/report.html`. It needs no network connection. The
joint `assignments.csv` preserves the source sample and biological-unit IDs;
each arm also exports its assignments and maximum conditional membership.
Cluster numbers are arbitrary within each arm. Agreement does not select a
winning model or establish species identity.

For just one neural arm:

```bash
super-deli cluster-genomic-neural examples/genomic-neural/dataset.json \
  --model-config examples/genomic-neural/model.json \
  --output-dir genomic-neural-01
```

Every output directory must be new. Failures and partial numerical evidence
are retained, with no replacement seed or substitution of one cluster.

## Declare your data

Use the existing dataset schema with a specimen-level genomic modality, an
explicit observation-to-specimen map and `format` set to `vcf`, `phylip`,
`structure` or `genepop`. SNPio reads the source into a `GenotypeData` object.
Use the existing explicit reader options for STRUCTURE/GENEPOP allele encoding.
`popmapfile` may be omitted or set to `null`. Population metadata is retained
for provenance and never becomes a predictor or training label.

The new workflow requires complete supported diploid biallelic genotypes,
one observation per biological specimen and an explicit independence assumption.
Missing, unsupported and multiallelic calls are rejected before fitting; the
older genomic workflow's missing-data policy remains available there. IUPAC
input cannot establish original ploidy, which remains a user declaration.
Monomorphic loci are recorded and removed within each training fit.

The neural model declaration requires schema 1, `ploidy: 2`,
`missing_policy: "error"` and `assume_independent_specimens: true`.
The comparison declaration wraps that object as `model`, specifies schema 1 and
lists every prespecified `neural_seeds` value. No best-seed selection occurs.
The explicit original-panel limit defaults to 512 loci and can be declared
between 2 and 4,096; this first CPU baseline is not a whole-genome-scale model.
Workspace and pairwise-stability budgets have separate explicit limits.
These bound the numerical workflow, not all memory used internally by SNPio.

## What is fitted

All arms use exactly one immutable source snapshot and one SNPio encoding with
a separate fixed encoding seed. Source file hashes, raw allele calls, original
locus indices and encoded dosages remain in the local evidence directory.

* **Original PCA/diagonal GMM:** unchanged Patterson scaling, PCA and native
  mixture-seed derivation, with locus bootstrap disabled in this coordinator.
* **PCA/shared GMM:** two PCA coordinates standardized on the training specimens,
  followed by one shared covariance across mixture components.
* **Categorical autoencoder/shared GMM:** Patterson-scaled inputs; encoder
  L–16–8–2; decoder 2–8–16–3L; tanh hidden layers and three genotype logits per
  retained locus. Mean categorical cross entropy reconstructs the observed
  0/1/2 calls. Fixed CPU float64 training uses 500 full-batch Adam updates,
  learning rate .001 and coupled weight decay .0001. Standardized latent
  coordinates enter the same shared-GMM search as the PCA control.

The new comparison defaults to K=1..4, minimum component size three and ten EM
starts, retaining the native convergence/eligibility policy. BIC is compared
within each representation, never between raw PCA and neural score spaces.
The diagonal baseline also retains its historical seed derivation. Thus the
PCA/shared arm is the closer representation control for the neural model.

Decoder probabilities describe training reconstruction. They are not calibrated
genotype uncertainty, missing-call imputation, an HWE model, or species support.
GMM memberships are conditional assignments in the selected fitted score space.
No neural genomic p-value or species-boundary test is implemented.

## Specimen subsampling

Set `n_subsamples` and `subsampling_seed` explicitly to request paired 80%
specimen subsets without replacement. All comparison arms receive identical
original sample identities on each draw. Variable-locus filtering, Patterson
scaling, PCA/neural training, coordinate scaling and GMM selection are refitted.
The original source panel and SNPio allele coding stay fixed. Absent specimens
receive no guessed labels. K frequencies, pair inclusion, positive coassignment,
conditional Monte Carlo precision and failed-draw bounds remain separate.

This evaluates specimen sensitivity conditional on the sampled loci and fixed
optimizer seed. It is **not** the older whole-locus-block bootstrap, a measure
of sensitivity to linked regions, or a probability that a cluster is a species.
All requested failures remain in the denominator/bounds; the summary becomes
partial if a requested subsample fails.

## Python API and frozen projection

```python
from super_deli.io.config import load_config, read_json
from super_deli.models.genomic_neural_config import GenomicComparisonConfig
from super_deli.workflows.genomic_neural import run_genomic_comparison

summary = run_genomic_comparison(
    load_config("dataset.json"),
    analysis_config=GenomicComparisonConfig.from_dict(read_json("analysis.json")),
    base_dir=".",
    output_dir="genomic-comparison-02",
)
```

`load_genomic_model(arm_directory)` loads checksummed numeric NPZ arrays without
pickle. Its `predict(dosages, locus_keys=...)` returns coordinates, memberships
and zero-based labels. Supply complete dosages in the **identical original
locus order and allele coding**, using the saved `model.locus_keys` only after
verifying that contract. These source-hash/index keys do not establish homology
between different files. There is no automatic raw-file harmonization or
projection CLI. A neural model's `reconstruct_probabilities` returns three
probabilities per retained locus; `arrays["kept_indices"]` maps them to source.

## Validation boundaries

The [bounded simulation comparison](genomic-neural-results.md) separates discrete
population recovery from continuous clines, admixture and linked-marker stress.
Only three independent main datasets per law/sample-size cell were planned.
Neural seeds and subsamples are repeated measurements, not extra datasets.
These results establish a checked baseline and operating examples, not general
false-split control, neural superiority or empirical species validation.
Empirical evidence gates and historical p-values remain unchanged.
