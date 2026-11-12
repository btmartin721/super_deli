# Opt-in reference-assignment abstention

This additive workflow keeps a fitted residual classifier's original probabilities
and reference labels, and separately decides whether to report an accepted label.
It supports complete continuous morphology and individually measured ecology from
the same independent specimens. Existing train/predict commands remain unchanged.
Use Python 3.11 or 3.12. Calibration and prediction need the NumPy-based clustering
extra; training the original residual classifier needs the CPU neural extra.

```bash
super-deli calibrate-reference-abstention calibration/dataset.json \
  --training-dataset training/dataset.json \
  --reference-run residual-training \
  --labels calibration/references.csv \
  --abstention-config examples/reference-abstention/config.json \
  --output-dir reference-support

super-deli predict-reference-abstention new-specimens/dataset.json \
  --reference-run residual-training \
  --calibration-run reference-support \
  --output-dir supported-assignments
```

The original residual model must be a complete, unchanged run. Original training
tables and labels are bound by hashes and specimen identity. Calibration labels
use the separate `unit_id,reference_label` CSV format and require exactly the
original reference classes, each with at least twenty calibration specimens.
Explicitly acknowledge independent calibration specimens. Specimens must not
cross training, calibration or prediction roles. Different IDs do not prove that
sampling is independent; callers remain responsible for the sampling design.

The Python API exposes `ReferenceAbstentionConfig`,
`calibrate_reference_abstention`, `load_reference_abstention` and
`predict_reference_abstention` in the corresponding models/workflows modules.
Saved models use strict JSON/NPZ, never pickle. Frozen prediction does not import
Torch or SNPio and retains the fitted feature order and training-only transforms.

## Compatible references and decisions

The `reference-assignments.csv` output retains `predicted_reference_label` and
the original class probabilities. The separate `accepted_reference_label` is blank
when the assignment is withheld. `compatible_references` lists the references
meeting the chosen support rule. `decision` distinguishes:

- `assigned`: a single compatible reference agrees with the classifier;
- `withheld_outside_reference_support`: no compatible reference;
- `withheld_ambiguous_reference_support`: more than one compatible reference;
- `withheld_classifier_support_conflict`: one compatible reference disagrees
  with the classifier's original prediction.

Conflicting labels are never silently substituted. Input or numerical failures
produce an unavailable failure record, not a successful abstention. A local HTML
report summarizes assignments. JSON/NPZ retain provenance and calibration evidence.

## Fixed support rules

Choose the policy in advance with `policy: distance` or `policy: probability`.
The classwise exclusion level is fixed at .05 for this version.

Distance support uses original training-standardized, equally view-balanced raw
traits. Fit one mean per known class and a pooled within-class covariance from
training specimens only. The covariance is `.9*S + .1*trace(S)/D*I`, with no
post-hoc variance floor. Unresolved covariance fits fail explicitly. Score each
reference with squared Mahalanobis distance. This use of a quadratic distance
does not assume a Gaussian species null or produce a species probability.

Probability support scores each class with `1-p_class`, retaining the original
classifier probabilities without temperature scaling or retraining.

Support uses only the classifier's retained features. Changes confined to dropped
or unmeasured features are not assessed.

For each class, sort scores from calibration specimens bearing that reference
label and take the `ceil((m+1)*.95)`-th score, indexed from one. A reference is
compatible when its score is at or below that threshold; ties are included.
Scores, class counts, ranks and thresholds remain available for audit.

With independent frozen training and suitable within-class exchangeability, this
construction supports a true-known-label inclusion statement for the compatible
set. It does not guarantee 95% accuracy among accepted assignments, <=5% unfamiliar
acceptance, validity under distribution shift, or detection of unfamiliar species.
The later singleton/agreement decision is distinct from set inclusion. Threshold
calibration here is separate from existing GMM fitted-null significance workflows;
no p-values or species-support probabilities are computed or modified.

An unsupported specimen may reflect measurement problems, changed conditions,
ordinary overlap or a missing reference population/species. An unfamiliar group
with the same measured distribution as a known reference cannot be reliably
distinguished from it using these measurements. Withholding is not species discovery.

The [simulation protocol](reference-abstention-validation.md) assesses these limits.
