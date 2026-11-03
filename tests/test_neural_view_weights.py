"""Weighted loss identities, default parity and numeric model roundtrips."""

# ruff: noqa: E402
from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("sklearn")

from super_deli.benchmarking.matched_audit import synthetic_tables
from super_deli.benchmarking.matched_data import configuration, write_dataset
from super_deli.benchmarking.neural_analysis import arrays
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.io.config import read_json, write_json
from super_deli.models.matched_neural import fit_matched_point
from super_deli.models.matched_preprocessing import prepare_views
from super_deli.workflows.matched_neural import load_matched_neural_model, run_matched_neural


def data():
    rng = np.random.default_rng(430903)
    return {
        "morphology": rng.normal(size=(32, 4)),
        "ecology": rng.normal(size=(32, 8)),
        "ecology_order": rng.permutation(32),
    }


@pytest.mark.parametrize("fractions", [(0.5, 0.5), (0.8, 0.2), (0.2, 0.8)])
def test_per_view_loss_identity_after_constant_exclusion(fractions):
    d = data()
    x = np.column_stack((d["morphology"], d["ecology"]))
    x[:, 0], x[:, 4:6] = 4, 3
    z, a = prepare_views(x, 4, fractions=fractions)
    np.testing.assert_array_equal(a["retained_view_counts"], [3, 6])
    expected = np.sqrt(9 * np.array(fractions) / [3, 6])
    np.testing.assert_allclose(a["view_weights"], expected, atol=1e-15)
    residual = np.arange(z.size).reshape(z.shape) / 100
    ordinary = residual / a["column_weights"]
    assert np.mean(residual**2) == pytest.approx(
        fractions[0] * np.mean(ordinary[:, :3] ** 2) + fractions[1] * np.mean(ordinary[:, 3:] ** 2)
    )


def test_equal_weights_keep_all_legacy_numeric_bits():
    d = data()
    x = np.column_stack((d["morphology"], d["ecology"]))
    c = configuration("paired_0", 0)
    legacy = fit_matched_point(x, config=c)
    weighted = fit_matched_point(
        x, config=replace(c, schema_version=2, view_loss_weights=(0.5, 0.5))
    )
    for key, value in legacy.model.arrays.items():
        np.testing.assert_array_equal(value, weighted.model.arrays[key])
    for name in ("losses", "scores", "labels", "strength"):
        np.testing.assert_array_equal(getattr(legacy, name), getattr(weighted, name))
    assert "declared_view_loss_weights" not in legacy.model.arrays


@pytest.fixture(scope="module")
def weighted(tmp_path_factory):
    root = tmp_path_factory.mktemp("weighted")
    d = data()
    ds = write_dataset(d, root / "raw")
    c = replace(
        configuration("paired_101", 430907, 2), schema_version=2, view_loss_weights=(0.8, 0.2)
    )
    result = run_matched_neural(ds, model_config=c, base_dir=root / "raw", output_dir=root / "fit")
    return root, d, result


def test_weighted_projection_reconstruction_and_subsamples(weighted):
    root, d, summary = weighted
    model = load_matched_neural_model(root / "fit")
    a = arrays(root / "fit/point.npz")
    np.testing.assert_array_equal(model.encode(synthetic_tables(d))["scores"], a["scores"])
    output = model.reconstruct(synthetic_tables(d))["views"]
    errors = [
        np.mean(((output[v] - d[v]) / a["scales"][s]) ** 2)
        for v, s in (("morphology", slice(0, 4)), ("ecology", slice(4, 12)))
    ]
    assert 0.8 * errors[0] + 0.2 * errors[1] == pytest.approx(a["losses"][-1], abs=1e-12)
    stable = arrays(root / "fit/stability.npz")
    x = np.column_stack((d["morphology"], d["ecology"]))
    for i, row in enumerate(read_json(root / "fit/subsamples.json")):
        ids = row["training_indices"]
        assert row["unit_ids"] == [f"unit_{j:04d}" for j in ids]
        np.testing.assert_allclose(stable[f"draw_{i}_means"], x[ids].mean(0), atol=1e-12)
        np.testing.assert_array_equal(stable[f"draw_{i}_normalized_view_loss_weights"], [0.8, 0.2])
    assert summary["species_significance"]["status"] == "not_computed"
    assert "80%" in (root / "fit/report.html").read_text()


def test_loader_rejects_self_consistent_but_wrong_fractions(tmp_path, weighted):
    import shutil

    root, _, _ = weighted
    target = tmp_path / "fit"
    shutil.copytree(root / "fit", target)
    a = arrays(target / "point.npz")
    a["declared_view_loss_weights"] = np.array([0.5, 0.5])
    np.savez_compressed(target / "point.npz", **a)
    summary = read_json(target / "summary.json")
    summary["model_sha256"] = fingerprint_file(target / "point.npz").sha256
    summary["configuration_sha256"] = configuration_hash(summary["configuration"])
    (target / "summary.json").unlink()
    write_json(summary, target / "summary.json")
    with pytest.raises(ValueError, match="view-loss"):
        load_matched_neural_model(target)
