"""Exact native branch parity, frozen projection, redundancy and retained failures."""

# ruff: noqa: E402
from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from super_deli.benchmarking.matched_audit import synthetic_tables
from super_deli.benchmarking.matched_data import configuration, write_dataset
from super_deli.benchmarking.neural_analysis import arrays
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json
from super_deli.models.separate_config import SeparateNeuralConfig
from super_deli.models.separate_neural import fit_separate_point, fusion_spectrum, unpack_branch
from super_deli.models.tabular_neural import fit_neural_point, point_arrays
from super_deli.workflows.separate_neural import load_separate_neural_model, run_separate_neural


def fixture_data():
    rng = np.random.default_rng(100151)
    return {
        "morphology": rng.normal(size=(32, 4)),
        "ecology": rng.normal(size=(32, 8)),
        "ecology_order": rng.permutation(32),
    }


def config(seed=0, draws=0):
    old = configuration("paired_" + str(seed), 100153, draws).to_dict()
    return SeparateNeuralConfig.from_dict(
        {**old, "model_family": "matched_separate_autoencoders_hdbscan"}
    )


@pytest.mark.parametrize("seed", [0, 101, 202])
def test_exact_standalone_branch_parity(seed):
    d = fixture_data()
    x = np.column_stack((d["morphology"], d["ecology"]))
    result = fit_separate_point(x, config=config(seed))
    expected = []
    for role in ("morphology", "ecology"):
        c = configuration(role + "_" + str(seed), 0)
        native = fit_neural_point(d[role], c.feature_names, config=c)
        actual = unpack_branch(result.model.arrays, role)
        for key, value in point_arrays(native).items():
            if key not in ("labels", "strength"):
                np.testing.assert_array_equal(actual[key], value)
        expected.append(native.scores)
    np.testing.assert_array_equal(result.scores, np.column_stack(expected))
    assert result.scores.shape == (32, 4)


def test_genuine_cross_view_redundancy_is_retained():
    x = np.random.default_rng(17).normal(size=(32, 2))
    singular, rank = fusion_spectrum(np.column_stack((x, x)))
    assert rank == 2 and singular.shape == (4,)
    with pytest.raises(ValueError):
        fusion_spectrum(np.ones((32, 4)))


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    root = tmp_path_factory.mktemp("separate")
    d = fixture_data()
    dataset = write_dataset(d, root / "raw")
    summary = run_separate_neural(
        dataset, model_config=config(101, 2), base_dir=root / "raw", output_dir=root / "fit"
    )
    return root, d, summary


def test_frozen_transforms_and_native_subsampling(fitted):
    root, d, summary = fitted
    model = load_separate_neural_model(root / "fit")
    point = arrays(root / "fit/point.npz")
    np.testing.assert_array_equal(model.encode(synthetic_tables(d))["scores"], point["scores"])
    reconstructed = model.reconstruct(synthetic_tables(d))["views"]
    for role in ("morphology", "ecology"):
        a = unpack_branch(point, role)
        mse = np.mean(
            (
                (reconstructed[role][:, a["kept_indices"]] - d[role][:, a["kept_indices"]])
                / a["scales"][a["kept_indices"]]
            )
            ** 2
        )
        assert mse == pytest.approx(a["losses"][-1], abs=1e-12)
    saved = arrays(root / "fit/stability.npz")
    for i, row in enumerate(read_json(root / "fit/subsamples.json")):
        ids = row["training_indices"]
        assert row["unit_ids"] == [f"unit_{j:04d}" for j in ids]
        for role in ("morphology", "ecology"):
            np.testing.assert_allclose(
                saved[f"draw_{i}_{role}__means"], d[role][ids].mean(0), atol=1e-12
            )
    assert (
        summary["n_coordinates"] == 4
        and summary["species_significance"]["status"] == "not_computed"
    )
    import csv

    with (root / "fit/assignments.csv").open() as handle:
        rows = list(csv.reader(handle))
    assert len(rows[0]) == 9 and all(len(row) == 9 for row in rows)
    assert rows[0][-1] == "latent_4"
    assert "all four" in (root / "fit/report.html").read_text()


def test_self_consistent_hash_does_not_hide_wrong_coordinate_order(tmp_path, fitted):
    import shutil

    root, _, _ = fitted
    target = tmp_path / "fit"
    shutil.copytree(root / "fit", target)
    a = arrays(target / "point.npz")
    a["scores"] = a["scores"][:, [2, 3, 0, 1]]
    np.savez_compressed(target / "point.npz", **a)
    summary = read_json(target / "summary.json")
    summary["model_sha256"] = fingerprint_file(target / "point.npz").sha256
    (target / "summary.json").unlink()
    write_json(summary, target / "summary.json")
    with pytest.raises(ValueError, match="coordinate order"):
        load_separate_neural_model(target)


def test_working_budget_guard():
    with pytest.raises(ValueError, match="workspace"):
        fit_separate_point(np.ones((32, 12)), config=replace(config(), max_working_bytes=1))


def test_second_branch_failure_retains_first_and_partial(tmp_path, monkeypatch):
    from super_deli.models import tabular_neural

    original = tabular_neural._train
    calls = []

    def fail_second(z, seed):
        calls.append(z.shape[1])
        if len(calls) == 2:
            exc = ValueError("injected second-branch training failure")
            exc.arrays = {"partial_marker": np.array([17])}
            raise exc
        return original(z, seed)

    monkeypatch.setattr(tabular_neural, "_train", fail_second)
    d = fixture_data()
    dataset = write_dataset(d, tmp_path / "raw")
    out = tmp_path / "failed"
    with pytest.raises(ValueError, match="injected second-branch"):
        run_separate_neural(
            dataset, model_config=config(), base_dir=tmp_path / "raw", output_dir=out
        )
    retained = arrays(out / "failed-training.npz")
    assert calls == [4, 8]
    assert retained["morphology__losses"].shape == (501,)
    np.testing.assert_array_equal(retained["ecology__failed_partial_marker"], [17])
    assert read_json(out / "failure.json")["diagnostics"]["completed_branches"] == 1
    assert not (out / "assignments.csv").exists()
    assert not (out / "summary.json").exists()
