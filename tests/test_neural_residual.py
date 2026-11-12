"""Independent derivatives, actual optimizer path and frozen reference workflows."""

# ruff: noqa: E402

import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from super_deli.benchmarking.residual_config import model_for
from super_deli.benchmarking.residual_data import write_cohort, write_labels
from super_deli.benchmarking.residual_derivatives import evaluate
from super_deli.benchmarking.residual_observer import ResidualObserver
from super_deli.benchmarking.view_weight_observer import NumericalAuditFailure
from super_deli.models.residual_network import (
    fit_residual_point,
    forward_numpy,
    initial_weights,
    probabilities,
)
from super_deli.workflows.residual_reference import (
    load_residual_reference_model,
    predict_matched_residual,
    train_matched_residual,
)


def invented():
    rng = np.random.default_rng(8237)
    y = np.repeat(np.arange(4), 4)
    x = rng.normal(size=(16, 12))
    return dict(morphology=x[:, :4], ecology=x[:, 4:], truth=y, ecology_order=rng.permutation(16))


def test_residual_analytic_gradients_have_finite_difference_agreement():
    x = np.random.default_rng(8).normal(size=(8, 6))
    y = np.tile(np.arange(4), 2)
    w = initial_weights(6, 4, 10)
    loss, p, g = evaluate(x, y, w, differentiate=True)
    np.testing.assert_allclose(p, probabilities(forward_numpy(x, w)), rtol=1e-12, atol=1e-12)
    assert np.isfinite(loss)
    for name in w:
        for index in (0, w[name].size - 1):
            original = w[name].flat[index]
            w[name].flat[index] = original + 1e-6
            plus = evaluate(x, y, w)[0]
            w[name].flat[index] = original - 1e-6
            minus = evaluate(x, y, w)[0]
            w[name].flat[index] = original
            np.testing.assert_allclose(
                (plus - minus) / 2e-6, g[name].flat[index], rtol=2e-5, atol=2e-8
            )


def test_actual_path_and_frozen_predictions(tmp_path):
    data = invented()
    dataset, units = write_cohort(data, tmp_path / "raw", tag="invented")
    labels = tmp_path / "references.csv"
    write_labels(units, data["truth"], labels)
    c = model_for("residual", 0)
    observer = ResidualObserver()
    summary = train_matched_residual(
        dataset,
        labels_path=labels,
        model_config=c,
        base_dir=tmp_path / "raw",
        output_dir=tmp_path / "fit",
        _observer=observer,
    )
    model = load_residual_reference_model(tmp_path / "fit")
    audited = observer.finish(model.arrays)
    assert audited["errors"].shape == (500, 4)
    assert summary["training"]["parameter_count"] == 4772
    assert summary["species_significance"] == "not_computed"
    x = np.column_stack((data["morphology"], data["ecology"]))
    before = {k: v.copy() for k, v in model.arrays.items()}
    p = model.predict_matrix(x)["probabilities"]
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-12)
    model.predict_matrix(x * 3 + 20)
    for key in before:
        np.testing.assert_array_equal(before[key], model.arrays[key])
    prediction = predict_matched_residual(
        dataset,
        reference_run=tmp_path / "fit",
        base_dir=tmp_path / "raw",
        output_dir=tmp_path / "predict",
    )
    assert prediction["unit_ids"] == units
    code = (
        "import sys;from super_deli.workflows.residual_reference "
        "import load_residual_reference_model;"
        "load_residual_reference_model(sys.argv[1]);"
        "assert 'torch' not in sys.modules;assert 'snpio' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code, str(tmp_path / "fit")], check=True)
    path = tmp_path / "fit" / "model.npz"
    arrays = {**model.arrays, "input_weight": model.arrays["input_weight"] + 1}
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="fingerprint"):
        load_residual_reference_model(tmp_path / "fit")


def test_reference_failures_preserve_inputs_and_partial_evidence(tmp_path):
    data = invented()
    dataset, units = write_cohort(data, tmp_path / "raw", tag="negative")
    labels = tmp_path / "references.csv"
    labels.write_text("unit_id,reference_label\nwrong,a\n")
    with pytest.raises(ValueError):
        train_matched_residual(
            dataset,
            labels_path=labels,
            model_config=model_for("residual", 0),
            base_dir=tmp_path / "raw",
            output_dir=tmp_path / "failed",
        )
    assert (tmp_path / "failed/failure.json").exists()
    assert not (tmp_path / "failed/reference-predictions.csv").exists()
    assert labels.read_text() == "unit_id,reference_label\nwrong,a\n"


def test_injected_optimizer_failure_is_retained():
    class Broken(ResidualObserver):
        def pre(self, *args):
            raise NumericalAuditFailure("invented audit guard")

    data = invented()
    with pytest.raises(ValueError) as error:
        fit_residual_point(
            np.column_stack((data["morphology"], data["ecology"])),
            data["truth"],
            config=model_for("residual", 0),
            observer=Broken(),
        )
    assert error.value.diagnostics["completed_updates"] == 0
    assert error.value.arrays


@pytest.mark.parametrize(
    "y",
    [
        np.zeros(16, dtype=int),
        np.full(16, -1),
        np.tile([0, 2], 8),
        np.ones(16, dtype=float),
        np.zeros(15, dtype=int),
    ],
)
def test_invalid_encoded_labels(y):
    data = invented()
    with pytest.raises(ValueError):
        fit_residual_point(
            np.column_stack((data["morphology"], data["ecology"])),
            y,
            config=model_for("residual", 0),
        )
