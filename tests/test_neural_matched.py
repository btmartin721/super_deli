"""Paired identity safety, loss balance, frozen transforms and failure boundaries."""
# ruff: noqa: E402

import json
from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("sklearn")
from super_deli.benchmarking.matched_audit import synthetic_tables
from super_deli.benchmarking.matched_data import configuration, write_dataset
from super_deli.core.provenance import fingerprint_file
from super_deli.models.matched_neural import fit_matched_neural
from super_deli.models.matched_preprocessing import align_tables, prepare_views
from super_deli.workflows.matched_neural import load_matched_neural_model, run_matched_neural


def data(n=32):
    rng = np.random.default_rng(109003)
    return {
        "morphology": rng.normal(size=(n, 4)),
        "ecology": rng.normal(size=(n, 8)),
        "ecology_order": rng.permutation(n),
    }


def test_balanced_loss_and_constant_exclusions():
    d = data()
    x = np.column_stack((d["morphology"], d["ecology"]))
    x[:, 0] = 3
    x[:, 4:6] = 5
    weighted, a = prepare_views(x, 4)
    np.testing.assert_array_equal(a["retained_view_counts"], [3, 6])
    np.testing.assert_allclose(a["view_weights"], np.sqrt([9 / 6, 9 / 12]))
    residual = np.arange(weighted.size).reshape(weighted.shape) / 100
    raw = residual / a["column_weights"]
    assert np.mean(residual**2) == pytest.approx(
        0.5 * (np.mean(raw[:, :3] ** 2) + np.mean(raw[:, 3:] ** 2))
    )
    assert set(a["kept_indices"]).isdisjoint({0, 4, 5})


def test_permuted_rows_align_exactly():
    d = data()
    tables = synthetic_tables(d)
    c = configuration("paired_0", 109007)
    reverse = np.arange(31, -1, -1)
    eco = replace(
        tables[1],
        unit_ids=tuple(tables[1].unit_ids[i] for i in reverse),
        observation_ids=tuple(tables[1].observation_ids[i] for i in reverse),
        values=tuple(tables[1].values[i] for i in reverse),
    )
    x, manifest = align_tables((eco, tables[0]), c)
    np.testing.assert_array_equal(x, np.column_stack((d["morphology"], d["ecology"])))
    assert manifest["views"][1]["source_row_indices"] == reverse.tolist()


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "missing_unit",
        "population",
        "missing_cell",
        "bool_cell",
        "reference_role",
        "count_dtype",
    ],
)
def test_bad_inputs_rejected_before_training(change, monkeypatch):
    tables = list(synthetic_tables(data()))
    c = configuration("paired_0", 1)
    t = tables[1]
    if change == "duplicate":
        t = replace(t, unit_ids=(t.unit_ids[0],) * len(t.unit_ids))
    elif change == "missing_unit":
        t = replace(t, unit_ids=("other", *t.unit_ids[1:]))
    elif change == "population":
        t = replace(t, level="population")
    elif change in ("missing_cell", "bool_cell"):
        row = (None if change == "missing_cell" else True, *t.values[0][1:])
        t = replace(t, values=(row, *t.values[1:]))
    elif change == "reference_role":
        t = replace(t, features=(replace(t.features[0], role="covariate"), *t.features[1:]))
    else:
        t = replace(t, features=(replace(t.features[0], dtype="count"), *t.features[1:]))
    monkeypatch.setattr(
        "super_deli.models.matched_neural._train", lambda *a: pytest.fail("No training allowed")
    )
    with pytest.raises(ValueError):
        fit_matched_neural((tables[0], t), config=c)


@pytest.mark.parametrize("draws", [0, 1])
def test_support_budget_precedes_training(monkeypatch, draws):
    c = replace(configuration("paired_0", 1, draws), max_pairwise_bytes=1)
    monkeypatch.setattr(
        "super_deli.models.matched_neural._train", lambda *a: pytest.fail("No training allowed")
    )
    with pytest.raises(ValueError, match="budget"):
        fit_matched_neural(synthetic_tables(data()), config=c)


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    root = tmp_path_factory.mktemp("paired-model")
    d = data()
    ds = write_dataset(d, root / "raw")
    c = configuration("paired_0", 109009, 2)
    result = run_matched_neural(ds, model_config=c, base_dir=root / "raw", output_dir=root / "fit")
    return root, d, result


def test_projection_reconstruction_and_matched_subsamples(saved):
    root, d, summary = saved
    model = load_matched_neural_model(root / "fit")
    tables = synthetic_tables(d)
    from super_deli.benchmarking.neural_analysis import arrays

    a = arrays(root / "fit/point.npz")
    stable = arrays(root / "fit/stability.npz")
    np.testing.assert_array_equal(model.encode(tables)["scores"], a["scores"])
    y = model.reconstruct(tables)
    errors = [
        np.mean(((y["views"][name] - d[name]) / a["scales"][section]) ** 2)
        for name, section in (("morphology", slice(0, 4)), ("ecology", slice(4, 12)))
    ]
    assert 0.5 * sum(errors) == pytest.approx(a["losses"][-1], abs=1e-12)
    assert summary["species_significance"]["status"] == "not_computed"
    records = json.loads((root / "fit/subsamples.json").read_text())
    x = np.column_stack((d["morphology"], d["ecology"]))
    for i, r in enumerate(records):
        ids = r["training_indices"]
        assert r["unit_ids"] == [f"unit_{j:04d}" for j in ids]
        np.testing.assert_allclose(stable[f"draw_{i}_means"], x[ids].mean(0), atol=1e-12)
        assert len(ids) == int(0.8 * len(x))
    text = (root / "fit/report.html").read_text()
    assert "Matched morphology" in text and "reconstruction MSE" in text


def test_reconstruction_names_cannot_overwrite_ids(saved):
    from super_deli.models.matched_neural import MatchedNeuralModel

    root, d, _ = saved
    model = load_matched_neural_model(root / "fit")
    config = replace(model.config, morphology="unit_ids", ecology="views")
    renamed = MatchedNeuralModel(config, model.arrays)
    tables = synthetic_tables(d)
    tables = (
        replace(tables[0], modality_name="unit_ids"),
        replace(tables[1], modality_name="views"),
    )
    result = renamed.reconstruct(tables)
    assert result["unit_ids"] == [f"unit_{i:04d}" for i in range(32)]
    assert set(result["views"]) == {"unit_ids", "views"}
    assert result["views"]["unit_ids"].shape == (32, 4)
    assert result["views"]["views"].shape == (32, 8)


def test_independent_optimizer_rejects_corrupted_loss_history(saved):
    from super_deli.benchmarking.matched_reference import audit_optimizer, balanced_inputs
    from super_deli.benchmarking.neural_analysis import arrays
    from super_deli.models.tabular_neural import _network, _torch, _weights

    root, d, _ = saved
    a = arrays(root / "fit/point.npz")
    torch = _torch()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        initial = _weights(_network(12))
    x = balanced_inputs(np.column_stack((d["morphology"], d["ecology"])), a, True)
    result = audit_optimizer(x, a, initial)
    assert result["updates"] == 500 and result["max_loss_absolute_difference"] < 1e-8
    a["losses"][250] += 1e-4
    with pytest.raises(AssertionError):
        audit_optimizer(x, a, initial)


def test_numeric_load_guards(tmp_path, saved):
    import shutil

    from super_deli.benchmarking.neural_analysis import arrays

    root, _, _ = saved
    copy = tmp_path / "fit"
    shutil.copytree(root / "fit", copy)
    path = copy / "point.npz"
    a = arrays(path)
    a["scales"][0] = 0
    np.savez_compressed(path, **a)
    with pytest.raises(ValueError, match="hash"):
        load_matched_neural_model(copy)
    summary = json.loads((copy / "summary.json").read_text())
    summary["model_sha256"] = fingerprint_file(path).sha256
    (copy / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="scales"):
        load_matched_neural_model(copy)


def test_small_subsample_failure_is_not_replaced():
    d = data(10)
    point, stability, _ = fit_matched_neural(
        synthetic_tables(d), config=configuration("paired_101", 109011, 1)
    )
    assert len(point.labels) == 10 and len(stability.records) == 1
    assert (
        stability.records[0]["status"] == "failed"
        and len(stability.records[0]["training_indices"]) == 8
    )
    assert stability.summary["successful"] == 0 and stability.summary["status"] == "unavailable"
    assert not np.any(stability.arrays["pair_successful"])


@pytest.mark.parametrize(
    "family,expected",
    [
        (
            "tabular_autoencoder_hdbscan",
            "b5969f76c0e01904fdb4d691aae51226bc109428ff1cf939b9641d86e319693c",
        ),
        (
            "tabular_denoising_autoencoder_hdbscan",
            "5d17f8516ba61ad1b654b89fec375a6aa8700bb301d3317e432093b44d6ef3b8",
        ),
    ],
)
def test_legacy_report_bytes_unchanged(tmp_path, family, expected):
    import hashlib
    from types import SimpleNamespace

    from super_deli.workflows._neural_common import _report

    # Goldens captured from cc3d759 before this helper gained optional view text.
    point = SimpleNamespace(
        scores=np.array([[-1.0, -2.0], [0.0, 1.0], [1.0, 2.0]]),
        labels=np.array([0, -1, 0]),
        strength=np.array([0.5, 0.0, 0.8]),
        losses=np.array([1.0, 0.5]),
    )
    summary = {
        "model": family,
        "selected_k": 1,
        "noise_count": 1,
        "stability": {"status": "not_computed"},
    }
    _report(tmp_path, summary, point, ("u1", "u<&2", "u3"))
    assert hashlib.sha256((tmp_path / "report.html").read_bytes()).hexdigest() == expected


def test_training_trace_is_read_only(tmp_path):
    from super_deli.benchmarking.matched_run import retain_fit_times, retain_initial_weights
    from super_deli.models import tabular_neural as numerical

    c = configuration("morphology_101", 109151)
    x = data()["morphology"]
    direct = numerical.fit_neural_point(x, c.feature_names, config=c)
    original_network = numerical._network
    original_fit = numerical.fit_neural_point
    with (
        retain_fit_times("morphology_101", tmp_path / "trace") as events,
        retain_initial_weights(tmp_path / "trace", events),
    ):
        traced = numerical.fit_neural_point(x, c.feature_names, config=c)
    assert numerical._network is original_network and numerical.fit_neural_point is original_fit
    a, b = numerical.point_arrays(direct), numerical.point_arrays(traced)
    assert set(a) == set(b)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
