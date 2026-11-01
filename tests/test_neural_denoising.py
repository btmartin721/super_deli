"""Independent corruption/gradient audits and legacy preservation checks."""

# ruff: noqa: E402
from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from super_deli.benchmarking.denoising_numerics import audit_training
from super_deli.benchmarking.neural_study import FEATURES, write_dataset
from super_deli.models.denoising_config import DenoisingConfig
from super_deli.models.denoising_train import train_denoising
from super_deli.models.tabular_denoising import fit_denoising_point, fit_tabular_denoising
from super_deli.models.tabular_neural import NeuralFitError, _train, point_arrays
from super_deli.workflows.tabular_denoising import load_denoising_model, run_tabular_denoising


def test_zero_corruption_is_exact_legacy_parity():
    x = np.random.default_rng(771).normal(size=(16, 4))
    old, losses = _train(x, 7)
    new, clean, _ = train_denoising(x, 7, 44, _corruption_sd=0)
    np.testing.assert_array_equal(clean, losses)
    for key in old:
        np.testing.assert_array_equal(new[key], old[key])


def test_corruption_audit_reproducibility_and_global_state():
    x = np.random.default_rng(772).normal(size=(20, 4))
    names = ("a", "b", "c", "d")
    config = DenoisingConfig(names, True, random_seed=7, corruption_seed=8)
    state = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    a = fit_denoising_point(x, names, config=config)
    b = fit_denoising_point(x, names, config=config)
    assert torch.equal(state, torch.random.get_rng_state())
    assert threads == torch.get_num_threads()
    for key, value in point_arrays(a).items():
        np.testing.assert_array_equal(value, point_arrays(b)[key])
    z = (x - x.mean(0)) / x.std(0)
    arrays = point_arrays(a)
    audit_training(z, arrays, 8)
    damaged = dict(arrays, corrupted_objectives=arrays["corrupted_objectives"].copy())
    damaged["corrupted_objectives"][0] += 0.01
    with pytest.raises(AssertionError):
        audit_training(z, damaged, 8)
    with pytest.raises(ValueError, match="digest"):
        audit_training(z, arrays, 9)
    np.testing.assert_allclose(a.model.encode(x), a.scores, atol=1e-12)
    assert not hasattr(a.model, "predict")


def test_failed_update_restores_rng_and_retains_evidence(monkeypatch):
    before = torch.random.get_rng_state().clone()
    deterministic = torch.are_deterministic_algorithms_enabled()
    threads = torch.get_num_threads()

    def fail(*a, **kw):
        raise ValueError("forced update failure")

    monkeypatch.setattr(torch.optim.Adam, "step", fail)
    with pytest.raises(NeuralFitError, match="forced") as caught:
        train_denoising(np.ones((12, 4)), 1, 2)
    assert caught.value.diagnostics["completed_updates"] == 0
    assert len(caught.value.diagnostics["corruption_sha256"]) == 1
    assert "checkpoint_0_encoder_0_weight" in caught.value.arrays
    assert torch.equal(before, torch.random.get_rng_state())
    assert threads == torch.get_num_threads()
    assert deterministic == torch.are_deterministic_algorithms_enabled()


def test_native_subsampling_roundtrip_and_invalid_tables(tmp_path, monkeypatch):
    from super_deli.io.tabular import ingest_tabular
    from super_deli.models import tabular_denoising as module

    raw = np.random.default_rng(773).normal(size=(24, 8))
    dataset = write_dataset(raw, tmp_path / "raw")
    config = DenoisingConfig(FEATURES, True, n_subsamples=2, corruption_seed=4)
    out = tmp_path / "fit"
    result = run_tabular_denoising(
        dataset, model_config=config, base_dir=tmp_path / "raw", output_dir=out
    )
    assert result["stability"]["successful"] == 2
    assert result["species_significance"]["status"] == "not_computed"
    np.testing.assert_allclose(
        load_denoising_model(out).encode(raw), np.load(out / "point.npz")["scores"], atol=1e-12
    )
    assert "denoising autoencoder" in (out / "report.html").read_text()
    table = ingest_tabular(
        dataset, base_dir=tmp_path / "raw", output_dir=tmp_path / "ingest"
    ).tables[0]
    with pytest.raises(ValueError, match="Repeated"):
        fit_tabular_denoising(replace(table, unit_ids=("same",) * 24), config=config)
    original = module.fit_denoising_point
    count = []

    def fit(x, names, *, config):
        count.append(len(x))
        if len(count) == 2:
            raise NeuralFitError("forced draw", arrays={"partial": np.ones(1)})
        return original(x, names, config=config)

    monkeypatch.setattr(module, "fit_denoising_point", fit)
    _, support = fit_tabular_denoising(table, config=config)
    assert support.summary["failed"] == 1
    assert "draw_0_failed_partial" in support.arrays
    assert support.records[0]["corruption_seed"] == 4
    assert np.any(support.arrays["coassignment_upper"] > support.arrays["coassignment_lower"])


@pytest.mark.parametrize("x", [np.ones((20, 4)), np.full((20, 4), np.nan), np.ones((9, 4))])
def test_bad_information(x):
    with pytest.raises(ValueError):
        fit_denoising_point(
            x, ("a", "b", "c", "d"), config=DenoisingConfig(("a", "b", "c", "d"), True)
        )


def test_significance_rejected_without_output(tmp_path):
    from super_deli.io.config import write_json
    from super_deli.models.calibration_config import GaussianCalibrationConfig
    from super_deli.workflows.score_calibration import run_integrative_calibration

    source = tmp_path / "source"
    source.mkdir()
    write_json({"model": "tabular_denoising_autoencoder_hdbscan"}, source / "summary.json")
    with pytest.raises(ValueError, match="Neural significance"):
        run_integrative_calibration(
            source, config=GaussianCalibrationConfig(True), output_dir=tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()
