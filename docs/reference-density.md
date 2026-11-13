# Opt-in known-only class-specific reference support

This separately versioned support layer fits complete, continuous matched
individual morphology/ecology using known references only. It accepts or
withholds the original residual classifier prediction. It discovers no new
reference class and computes no species p-value. Existing defaults, policies,
serialization and historical p-values are unchanged.

The existing distance policy uses one pooled within-reference covariance; local
support uses fifth-neighbor distances in that same pooled metric. The new policy
uses one Gaussian mean/covariance per reference. Each ML covariance receives the
fixed regularization `0.9*S + 0.1*trace(S)/d*I`. This is part of the estimator,
not a post-fit variance-floor repair. Constant/unresolved class fits fail explicitly.
The original classifier's training-only means, scales, retained traits and
balanced view weights are frozen and reused. No PCA or neural latent coordinates
replace the measured support features.

A separate known-only calibration set provides each class's 95% rank cutoff:
`ceil((m+1)*0.95)`, with at least twenty calibration specimens per reference.
At the scientific study's forty/class, the retained rank is 39. Class-specific
Gaussian log-density is a monotone function of squared Mahalanobis distance
within a class: the log determinant cancels in separate classwise rank cutoffs.
This policy therefore uses class-specific Gaussian ellipsoids; it does not add
a separate determinant-based ranking advantage.

Under independent classwise exchangeability, the nominal level concerns inclusion
of a known specimen's true reference in the compatible set. It does not guarantee
unique accepted assignment, accuracy among accepted assignments, rejection of
unfamiliar groups or coverage under environmental/sampling shifts. The three
withheld states remain distinct: no compatible reference, multiple compatible
references, and one compatible reference conflicting with the classifier argmax.
No unknown examples, evaluation labels, artificial outliers or threshold search
enter calibration.

Train a residual reference model with the existing interface, then calibrate:

```bash
super-deli calibrate-reference-density calibration.json \
  --training-dataset training.json \
  --reference-run reference-fit \
  --labels calibration-labels.csv \
  --support-config examples/known-reference/support.json \
  --output-dir density-calibration

super-deli predict-reference-density query.json \
  --reference-run reference-fit \
  --calibration-run density-calibration \
  --output-dir density-predictions
```

The training table/IDs/labels must match the original reference fit. Calibration
must contain exactly those known classes and use disjoint specimen identities.
Prediction identities must be disjoint from both. Labels are separate
`unit_id,reference_label` tables and never analytical features. Use the existing
[residual interface](residual-reference.md) and dataset schema for input formats.

The API is `ReferenceDensityConfig` in `models.reference_density_config`, with
`calibrate_reference_density`, `load_reference_density`, and
`predict_reference_density` in `workflows.reference_density`. Loading/prediction
use numeric artifacts without Torch or pickle. Full binding checks retain
training/calibration identities, original raw-source/model fingerprints, class
maps, covariance/count/rank schemas and finite numeric values. Frozen loading
recomputes support and calibration from retained known features and checks pooled
geometry against the unchanged predecessor binding.

`reference-assignments.csv` preserves the original predicted label and probability
columns, an optional accepted label, compatible references and an explicit decision.
`support-predictions.npz` retains numeric scores/decisions; `report.html` explains
the scope. Failures retain `failure.json` without a completed summary. These
files contain specimen information and should stay in the user's chosen local
analysis directory.

See the [held-out protocol](known-reference-validation.md). Empirical sampling and
reference-evidence gates remain unchanged; this checkpoint runs no empirical fit.
