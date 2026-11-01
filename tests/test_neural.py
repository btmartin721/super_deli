"""Independent forward/gradient checks and explicit inference/support boundaries."""

# ruff: noqa: E402
import json
from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from super_deli.core.schema import config_from_dict
from super_deli.io.config import write_json
from super_deli.models.neural_config import NeuralConfig
from super_deli.models.tabular_neural import (
    NeuralFitError,
    _network,
    _weights,
    fit_neural_point,
    fit_tabular_neural,
    forward_numpy,
    latent_transform,
)
from super_deli.workflows.tabular_neural import load_neural_model, run_tabular_neural

NAMES = ("a", "b", "c", "d")


def fixture(tmp_path, *, kind="morphological"):
    import csv

    raw = np.random.default_rng(551).normal(size=(32, 4))
    ids = [f"invented_{i}" for i in range(len(raw))]
    path = tmp_path / "traits.csv"
    with path.open("x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["observation_id", *NAMES])
        w.writerows([u, *r] for u, r in zip(ids, raw, strict=True))
    dataset = config_from_dict(
        {
            "schema_version": 1,
            "dataset_id": "invented_neural",
            "specimens": [{"specimen_id": u} for u in ids],
            "modalities": [
                {
                    "name": "traits",
                    "kind": kind,
                    "level": "specimen",
                    "format": "csv",
                    "path": "traits.csv",
                    "features": [{"name": n, "dtype": "continuous"} for n in NAMES],
                    "observations": [{"observation_id": u, "unit_id": u} for u in ids],
                }
            ],
        }
    )
    return raw, dataset


def test_gradients_match_finite_differences_and_numpy():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        net = _network(3)
    x = torch.tensor([[0.2, -0.1, 0.4], [0.5, 0.3, -0.2]], dtype=torch.float64)
    objective = ((net["decoder"](net["encoder"](x)) - x) ** 2).mean()
    objective.backward()
    for layer in (net["encoder"][0], net["decoder"][4]):
        p = layer.weight
        analytic = float(p.grad[0, 0])
        original = float(p.detach()[0, 0])
        vals = []
        for delta in (1e-6, -1e-6):
            with torch.no_grad():
                p[0, 0] = original + delta
                vals.append(float(((net["decoder"](net["encoder"](x)) - x) ** 2).mean()))
        with torch.no_grad():
            p[0, 0] = original
        assert analytic == pytest.approx((vals[0] - vals[1]) / 2e-6, abs=2e-9)
    weights = _weights(net)
    predicted = forward_numpy(forward_numpy(x.numpy(), weights, "encoder"), weights, "decoder")
    np.testing.assert_allclose(predicted, net["decoder"](net["encoder"](x)).detach(), atol=1e-14)


def test_determinism_rng_threads_losses_and_frozen_predictions():
    x = np.random.default_rng(3).normal(size=(32, 4))
    saved = x.copy()
    config = NeuralConfig(NAMES, True, random_seed=73)
    before = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    a = fit_neural_point(x, NAMES, config=config)
    b = fit_neural_point(x, NAMES, config=config)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert len(a.losses) == 501 and a.losses[-1] < a.losses[0]
    for key in a.model.arrays:
        np.testing.assert_array_equal(a.model.arrays[key], b.model.arrays[key])
    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_array_equal(x, saved)
    np.testing.assert_allclose(a.model.encode(x), a.scores, atol=1e-13)
    standardized = (x - a.model.arrays["means"]) / a.model.arrays["scales"]
    predicted = forward_numpy(
        forward_numpy(standardized, a.model.arrays, "encoder"), a.model.arrays, "decoder"
    )
    assert np.mean((predicted - standardized) ** 2) == pytest.approx(a.losses[-1], abs=1e-13)
    assert not hasattr(a.model, "predict")
    with pytest.raises(ValueError, match="complete rows"):
        a.model.encode([[np.nan] * 4])


@pytest.mark.parametrize("raw", [np.ones((20, 4)), np.full((20, 4), np.nan), np.ones((9, 4))])
def test_information_guards(raw):
    with pytest.raises(ValueError):
        fit_neural_point(raw, NAMES, config=NeuralConfig(NAMES, True))


def test_degenerate_embeddings_fail():
    for x in (np.zeros((12, 2)), np.column_stack([np.arange(12)] * 2), np.full((12, 2), np.inf)):
        with pytest.raises(NeuralFitError):
            latent_transform(x)


@pytest.mark.parametrize("kind", ["morphological", "ecological"])
def test_workflow_source_preservation_artifacts_and_roundtrip(tmp_path, kind):
    from super_deli.core.provenance import fingerprint_file

    raw, dataset = fixture(tmp_path, kind=kind)
    before = (tmp_path / "traits.csv").read_bytes()
    config = NeuralConfig(NAMES, True, n_subsamples=2, subsampling_seed=9)
    out = tmp_path / "run"
    s = run_tabular_neural(dataset, model_config=config, base_dir=tmp_path, output_dir=out)
    assert s["status"] == "complete" and s["stability"]["requested"] == 2
    assert s["stability"]["successful"] == 2
    assert (tmp_path / "traits.csv").read_bytes() == before
    for name, digest in s["artifact_sha256"].items():
        assert fingerprint_file(out / name).sha256 == digest
    model = load_neural_model(out)
    with np.load(out / "point.npz", allow_pickle=False) as f:
        np.testing.assert_allclose(model.encode(raw), f["scores"], atol=1e-13)
    assert model.reconstruct(raw).shape == raw.shape
    assert "species probability" in (out / "report.html").read_text()
    assert s["species_significance"]["status"] == "not_computed"
    with pytest.raises(FileExistsError):
        run_tabular_neural(dataset, model_config=config, base_dir=tmp_path, output_dir=out)


def test_failed_draw_keeps_identity_and_training_evidence(tmp_path, monkeypatch):
    from super_deli.io.tabular import ingest_tabular
    from super_deli.models import tabular_neural as m

    _, dataset = fixture(tmp_path)
    table = ingest_tabular(dataset, base_dir=tmp_path, output_dir=tmp_path / "ingest").tables[0]
    original = m.fit_neural_point
    calls = []

    def fit(x, names, *, config):
        calls.append(x.copy())
        if len(calls) == 2:
            raise NeuralFitError(
                "forced", diagnostics={"completed_updates": 2}, arrays={"checkpoint": np.ones(2)}
            )
        return original(x, names, config=config)

    monkeypatch.setattr(m, "fit_neural_point", fit)
    point, stability = fit_tabular_neural(table, config=NeuralConfig(NAMES, True, n_subsamples=2))
    assert len(calls) == 3 and len(point.labels) == 32
    assert stability.summary["failed"] == 1
    assert stability.records[0]["diagnostics"]["completed_updates"] == 2
    assert "draw_0_failed_checkpoint" in stability.arrays
    for record in stability.records:
        assert len(record["unit_ids"]) == len(set(record["unit_ids"])) == 25
    assert np.any(stability.arrays["coassignment_upper"] > stability.arrays["coassignment_lower"])


def test_point_failure_is_retained(tmp_path, monkeypatch):
    from super_deli.models import tabular_neural as m

    _, dataset = fixture(tmp_path)

    def fail(*args, **kwargs):
        raise NeuralFitError(
            "forced", diagnostics={"completed_updates": 4}, arrays={"checkpoint": np.arange(3)}
        )

    monkeypatch.setattr(m, "_train", fail)
    out = tmp_path / "failed"
    with pytest.raises(NeuralFitError, match="forced"):
        run_tabular_neural(
            dataset, model_config=NeuralConfig(NAMES, True), base_dir=tmp_path, output_dir=out
        )
    assert not (out / "summary.json").exists()
    assert json.loads((out / "failure.json").read_text())["diagnostics"]["completed_updates"] == 4
    assert (out / "failed-training.npz").is_file()


def test_constant_exclusion_and_explicit_noise(tmp_path, monkeypatch):
    from super_deli.models import tabular_neural as m

    raw = np.random.default_rng(9).normal(size=(24, 4))
    raw[:, 0] = 3
    monkeypatch.setattr(
        m,
        "fit_density_scores",
        lambda scores, config: (np.full(len(scores), -1), np.zeros(len(scores))),
    )
    point = fit_neural_point(raw, NAMES, config=NeuralConfig(NAMES, True))
    assert point.selected_k == 0 and np.all(point.labels == -1)
    assert point.diagnostics["excluded_constant_features"] == ["a"]
    assert np.all(point.model.reconstruct(raw)[:, 0] == 3)


def test_repeated_ids_and_unsupported_types_rejected(tmp_path):
    from super_deli.io.tabular import ingest_tabular

    _, dataset = fixture(tmp_path)
    table = ingest_tabular(dataset, base_dir=tmp_path, output_dir=tmp_path / "ingest").tables[0]
    repeated = replace(table, unit_ids=(table.unit_ids[0],) * len(table.unit_ids))
    with pytest.raises(ValueError, match="Repeated"):
        fit_tabular_neural(repeated, config=NeuralConfig(NAMES, True))
    features = (replace(table.features[0], dtype="count"), *table.features[1:])
    values = tuple((i % 5, *row[1:]) for i, row in enumerate(table.values))
    with pytest.raises(ValueError, match="continuous"):
        fit_tabular_neural(
            replace(table, features=features, values=values), config=NeuralConfig(NAMES, True)
        )


def test_configuration_is_saved_separately_from_reference_labels(tmp_path):
    _, dataset = fixture(tmp_path)
    config = NeuralConfig(NAMES, True)
    write_json(dataset.to_dict(), tmp_path / "dataset.json")
    write_json(config.to_dict(), tmp_path / "model.json")
    from super_deli.cli import main

    assert (
        main(
            [
                "cluster-tabular-neural",
                str(tmp_path / "dataset.json"),
                "--model-config",
                str(tmp_path / "model.json"),
                "--output-dir",
                str(tmp_path / "cli"),
            ]
        )
        == 0
    )


def test_neural_score_calibration_is_rejected_before_fit(tmp_path):
    from super_deli.models.calibration_config import GaussianCalibrationConfig
    from super_deli.workflows.score_calibration import run_integrative_calibration

    source = tmp_path / "source"
    source.mkdir()
    write_json({"model": "tabular_autoencoder_hdbscan"}, source / "summary.json")
    with pytest.raises(ValueError, match="Neural significance"):
        run_integrative_calibration(
            source, config=GaussianCalibrationConfig(True), output_dir=tmp_path / "calibration"
        )
    assert not (tmp_path / "calibration").exists()


def test_rng_and_threads_restore_after_optimizer_failure(monkeypatch):
    from super_deli.models.tabular_neural import _train

    state = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()

    def fail(*args, **kwargs):
        raise ValueError("forced optimizer error")

    monkeypatch.setattr(torch.optim.Adam, "step", fail)
    with pytest.raises(NeuralFitError, match="forced optimizer") as caught:
        _train(np.ones((12, 4)), 3)
    assert caught.value.diagnostics["completed_updates"] == 0
    assert caught.value.arrays
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
