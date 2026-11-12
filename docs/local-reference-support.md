# Class-local reference support

This opt-in workflow calibrates a fifth-nearest-training-specimen support score
for the existing residual reference classifier. It accepts complete continuous
morphology/ecology from matched independent individuals, with separately supplied
reference labels and independent known-class calibration specimens.

Install Python 3.11/3.12 and the neural extra to train the original classifier.
Frozen reference/support prediction needs NumPy and does not import Torch or SNPio.

```bash
super-deli calibrate-local-reference-support calibration/dataset.json \
  --training-dataset training/dataset.json --reference-run residual-model \
  --labels calibration/references.csv \
  --support-config examples/local-reference-support/config.json \
  --output-dir local-support
super-deli predict-local-reference-support prediction/dataset.json \
  --reference-run residual-model --calibration-run local-support \
  --output-dir local-prediction
```

The training declarations, source bytes and identities must match the original
classifier. Require at least five distinct original training unit IDs/reference,
at least twenty independent calibration specimens/reference, exactly the same
known classes and no training/calibration/prediction identity overlap. Equal
measurement values remain valid; disjoint IDs cannot establish biological
independence. The acknowledgement in the config is an assumption, not a validation.

The training-only standardization, view weights and .1-shrunk pooled covariance
remain those of the preceding distance policy. Each class uses the distance to its
fifth nearest training specimen in that fixed Mahalanobis metric. Calibration
supplies a separate classwise .95 rank threshold; it never becomes a neighbor.

`reference-assignments.csv` preserves the original classifier label/probabilities,
accepted label (blank when withheld), compatible references, their count and a
reason: assigned, outside all reference support, ambiguous support or classifier
conflict. A unique compatible class cannot silently replace the classifier argmax.
Raw scores, neighbor indices and tie counts are retained in numeric NPZ artifacts.
Index provenance refers to the original training-unit order in calibration metadata.
The HTML report explains the outputs. A completion summary is written last; failed
or incomplete workflows remain unavailable, not biological abstentions.

The new schema/commands do not reinterpret older calibration artifacts. Model,
input, class and feature hashes bind every calibration; loaders reject tampering,
inconsistent libraries/counts and unresolved covariance. Working/expanded-artifact
limits are enforced. No pickle or data-dependent fallback is used.

Nominal calibration concerns inclusion of a true known reference under appropriate
within-class exchangeability and a frozen independent training fit. It does not
certify accuracy among accepted specimens, unfamiliar-group rejection, domain-shift
performance, species probabilities or species boundaries. All defaults and existing
p-values remain unchanged. See the [protocol](local-reference-validation.md).

Python API: `calibrate_local_reference_support`, `load_local_reference_support` and
`predict_local_reference_support` in `super_deli.workflows.local_reference_support`;
configuration is `LocalReferenceSupportConfig` in `super_deli.models.local_reference_config`.

Run the explicitly invented installation check outside your analysis inputs:

```bash
python examples/local-reference-support/smoke.py --output-dir invented-local-support
```
