# Fit-free genomic control diagnosis: no-go for promotion

The prespecified decision is **no-go for promoting, broadly tuning or launching a
large calibration campaign for the current categorical autoencoder/shared-GMM**.
The existing model remains available as an experimental option. No default,
historical p-value, model trainer or prior artifact changed.

## Evidence and decision

| Neural seed | Control splits / 12 | PCA/shared splits / 12 | Neural-only splits | Improved ARI on six four-group datasets |
|---|---:|---:|---:|---:|
| neural_seed_0 | 12 | 1 | 11 | 0 |
| neural_seed_1 | 9 | 1 | 8 | 0 |
| neural_seed_2 | 12 | 1 | 11 | 0 |

The twelve controls combine common/rare-frequency HWE and N=64/128: these are
paired descriptive totals across heterogeneous conditions, not one estimated
false-positive rate. Each law/N cell still has only three independent datasets.
All five arms recovered all six strongly separated four-group datasets with
ARI=1. This easy case demonstrates no neural advantage; the audit does not rule
out benefits on untested problems or other neural architectures.

The auditor read all 48 archived observations, 240 point maps and 360 subsample
maps, including 360 neural maps. An eight-case resource preflight separately
evaluated 240 of the same maps. These are repeated audit evaluations, not fresh
datasets or performance replicates. No new network training, GMM search,
resampling, simulation or significance test was performed by the diagnosis.
The original 180,000 optimizer updates were linked through their archived audit
hashes/receipts, without replaying training. Ordinary engineering regressions
exercise existing workflows separately from this fit-free scientific analysis.

## What the audit establishes

* Consumed files match the sealed source manifest. Source identities, allele
  coding, retained loci, Patterson scaling, frozen coordinates and decoder
  losses agree with their saved values.
* Selected Gaussian densities/memberships, candidate eligibility, BIC ranking
  and parameter-cost accounting agree. Neural control splits are not explained
  by an audited selection or projection arithmetic error.
* The saved multi-component likelihood gains exceed their BIC parameter costs
  in split cases. This establishes why the existing selector chose those
  components in those coordinates, not a biological species count.
* Neural coordinate moments differ from Gaussian-shaped clouds. These are
  descriptive finite-sample moments, not normality tests or proof of separate
  modes. Hidden saturation and rare-variant associations vary by seed, layer
  and data law; they do not establish a universal causal explanation.
* In rare-frequency N=64 controls, rare-locus local sensitivity shares average
  about .906–.912 for neural seeds and about .919 for the PCA controls. The
  diagnostic does not support a claim of uniquely increased rare-locus
  sensitivity in the neural model. The share concerns local derivatives of
  the full scaled map, not global feature importance or a causal intervention.

The broader scientific mechanism remains **partly unresolved**. The evidence is
consistent with a learned non-Gaussian coordinate distribution being represented
by several Gaussian components. This is an interpretation of geometry and
existing fit costs; the fit-free audit cannot isolate training-objective,
representation and density-family causes through counterfactual experiments.

## Frozen descriptive measures

Training-set minor-allele dosage, heterozygote fraction, minor-allele carrier count
at MAF <=.1 and Patterson row energy were compared with coordinate radius/axes.
Spearman coefficients have no computed p-values. Label-associated variance is
conditional on the selected partition and unadjusted for K; it cannot validate
that partition independently. Tanh saturation uses absolute activation >=.95.
Analytic encoder Jacobians are checked by central finite differences on existing
map inputs, with epsilon 1e-6, rtol 1e-5 and atol 2e-7. These map evaluations do
not generate new biological observations. Sensitivity uses raw-dose units after
the saved input and coordinate scaling. PCA and neural score spaces retain their
own native BIC accounting; absolute BIC is never compared between spaces.

All 840 preflight/main map evaluations were separately checked with independent
dense Gaussian formulas, direct burden calculations and an independent tie-rank
correlation implementation. Saved subset identities and pair denominators remain
linked to original individuals. Source failures would withhold the scientific
decision; invented fixtures explicitly cannot produce a scientific decision.

## Development stopping decision

Keep PCA as the established comparison and retain the neural option as experimental.
This checkpoint authorizes no numerical repair: the audited arithmetic agrees.
Do not use an association with rare variants or saturation alone to select a
new loss, threshold, architecture or seed. Future development requires a separately
approved, prespecified advantage/control question and independent validation.
The present audit supplies no species-boundary guarantee, empirical validation,
general error-rate bound or universal rejection of neural approaches.

Full per-map evidence remains local. The [public export](../examples/genomic-control-diagnosis/results/diagnosis.json)
contains aggregates only. Prior work, Python 3.11/3.12 support and existing
significance workflows are preserved. No author was contacted.
