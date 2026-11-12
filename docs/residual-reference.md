# Supervised residual reference assignment

This opt-in model assigns complete, matched morphology/ecology measurements to
reference classes supplied during training. It is a supervised classifier;
it does not estimate the number of species or discover unfamiliar classes.
Reference taxonomy and independent sampling are user-supplied assumptions.
Existing clustering models, defaults and p-values are unchanged.

## Installation and commands

Use Python 3.11 or 3.12 and install `super-deli[neural]`. Training requires
PyTorch; frozen numerical prediction does not import Torch or SNPio.

```bash
super-deli train-matched-residual dataset.json \
  --labels references.csv \
  --model-config examples/residual-network/model.json \
  --output-dir reference-fit

super-deli predict-matched-residual new-individuals.json \
  --reference-run reference-fit \
  --output-dir reference-predictions
```

Every output directory must be new. The label CSV contains exactly
`unit_id,reference_label`, with one row for every training specimen and no
unmatched or duplicate IDs. Class names are nonempty, case-sensitive strings.
At least two specimens/class and 2–32 classes are required; this input minimum
does not establish adequate biological sampling.

Declare individually measured ecology and continuous morphology at specimen level,
with complete matching IDs, at least ten training specimens and 3–32 selected
traits per view. Three nonconstant traits per view must remain after training.
Labels are a separate source and cannot be selected as numerical features.
Prediction uses the fitted feature identities and class order; new observations
cannot change training means/scales or feature selection. Dataset acceptance
does not verify sampling independence or reference-label validity.

The API uses `ResidualConfig`, `train_matched_residual`,
`load_residual_reference_model` and `predict_matched_residual` in
`super_deli.models.residual_config` and
`super_deli.workflows.residual_reference`. The loaded model also exposes
`predict(tables)` and `predict_matrix(values)`, where matrix columns must follow
the fitted morphology-then-ecology feature order. Resource guards apply; large
cohorts can be predicted in batches without refitting.

## Fixed architecture

Training standardizes retained traits using training rows only and balances the
two views with the existing equal-view scaling rule. A width-32 input layer is
followed by two width-32 residual blocks and a reference-class logits layer.
Each block is `ReLU(h + W2 ReLU(W1 h+b1)+b2)`; the input uses ReLU as well.
Twelve retained features/four classes give 4,772 trainable parameters.

The model uses deterministic CPU float64, full-batch multiclass cross-entropy,
500 Adam updates, learning rate .001, betas .9/.999, epsilon 1e-8 and coupled
weight decay .0001 on all parameters. Hidden weights are initialized by an
independent NumPy generator with standard deviation `sqrt(2/fan_in)`; the output
uses `sqrt(1/32)`. Biases are zero. There is no batch normalization, dropout,
epoch selection or automatic seed search. Completing 500 updates is not a
convergence guarantee. Data loss and regularization are recorded separately.

The architecture is a compact custom residual baseline, not a reproduction of
the models in [Gorishniy et al.](https://proceedings.neurips.cc/paper_files/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html).

## Outputs and interpretation

`model.npz` stores numeric weights, initial weights, fitted preprocessing and
loss traces. `summary.json` binds configuration, classes and input/model hashes.
`cohort.json`, `training.json`, a copied label CSV and an HTML report retain
training provenance. Frozen reload checks hashes, numeric shapes, finite values,
view weights, initial-state declarations and class ordering without pickle.

`reference-predictions.csv` contains specimen IDs, the predicted reference label,
explicit reference-only scope and one probability column per class. `classes.json`
defines that column order. Probabilities condition on the fitted classifier;
they are not calibrated species probabilities. Training assignments do not
measure independent generalization. The classifier has no unfamiliar-class
rejection rule: it can assign high confidence to an unsupported reference label.
Novelty is explicitly unassessed.

Input or fitting failures retain a failure record and available partial numeric
evidence. Failed predictions are not reported as completed assignments. Missing
traits/views, noncontinuous measurements, grouped/related sampling, genomic and
landmark inputs, semi-supervision and uncertainty calibration are excluded.

See the [nonlinear comparison protocol](residual-nonlinear.md). The supplied
installation example uses invented individuals only.
