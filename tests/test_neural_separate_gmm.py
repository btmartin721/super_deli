"""Numerical parity, failure retention, memberships and whole-ID refits."""

# ruff: noqa: E402

import csv
from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("sklearn")

from super_deli.benchmarking.matched_audit import synthetic_tables
from super_deli.benchmarking.matched_data import configuration, write_dataset
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json
from super_deli.models.separate_gmm import fit_shared_point
from super_deli.models.separate_gmm_config import FAMILY, SeparateGMMConfig
from super_deli.models.separate_neural import fit_separate_point
from super_deli.workflows.separate_gmm import load_separate_gmm_model, run_separate_gmm


def data():
    rng = np.random.default_rng(51001)
    return dict(
        morphology=rng.normal(size=(32, 4)),
        ecology=rng.normal(size=(32, 8)),
        ecology_order=rng.permutation(32),
    )


def config(seed=0, draws=0):
    return SeparateGMMConfig.from_dict(
        {**configuration("paired_" + str(seed), 51003, draws).to_dict(), "model_family": FAMILY}
    )


@pytest.mark.parametrize("seed", [0, 101, 202])
def test_exact_encoder_parity_and_conditional_membership(seed):
    d = data()
    x = np.column_stack((d["morphology"], d["ecology"]))
    c = config(seed)
    new = fit_shared_point(x, config=c)
    old = fit_separate_point(x, config=c.encoder_config())
    np.testing.assert_array_equal(new.scores, old.scores)
    for k, v in old.model.arrays.items():
        np.testing.assert_array_equal(new.model.arrays[k], v)
    np.testing.assert_array_equal(new.losses, old.losses)
    prediction = new.model.predict(synthetic_tables(d))
    np.testing.assert_allclose(prediction["membership"], new.model.arrays["membership"], atol=1e-8)
    np.testing.assert_array_equal(prediction["labels"], new.labels)
    np.testing.assert_allclose(prediction["membership"].sum(1), 1, atol=1e-12)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    root = tmp_path_factory.mktemp("shared-encoders")
    d = data()
    dataset = write_dataset(d, root / "raw")
    c = config(101, 2)
    summary = run_separate_gmm(
        dataset, model_config=c, base_dir=root / "raw", output_dir=root / "fit"
    )
    return root, d, c, summary


def test_saved_model_and_subsample_coverage(fitted):
    root, d, c, summary = fitted
    model = load_separate_gmm_model(root / "fit")
    assert model.config == c and summary["species_significance"]["status"] == "not_computed"
    assert summary["stability"]["requested"] == 2 and summary["stability"]["successful"] == 2
    records = read_json(root / "fit/subsamples.json")
    for row in records:
        assert len(row["training_indices"]) == 25 and len(set(row["unit_ids"])) == 25
        assert row["diagnostics"]["completed_updates"] == 1000
    with (root / "fit/assignments.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert "density_membership_strength" not in rows[0]
    assert "component_1_probability" in rows[0]
    html = (root / "fit/report.html").read_text()
    assert "Gaussian mixture components" in html and "Density strength" not in html
    tables = synthetic_tables(d)
    np.testing.assert_allclose(
        model.predict(tables)["membership"], model.arrays["membership"], atol=1e-8
    )


def test_invalid_covariance_is_rejected_beyond_fingerprints(fitted, tmp_path):
    root, _, _, _ = fitted
    import shutil

    shutil.copytree(root / "fit", tmp_path / "broken")
    with np.load(tmp_path / "broken/point.npz", allow_pickle=False) as f:
        a = {k: f[k] for k in f.files}
    a["mixture_covariance"] = np.zeros((4, 4))
    np.savez_compressed(tmp_path / "broken/point.npz", **a)
    summary = read_json(tmp_path / "broken/summary.json")
    summary["model_sha256"] = fingerprint_file(tmp_path / "broken/point.npz").sha256
    (tmp_path / "broken/summary.json").unlink()
    write_json(summary, tmp_path / "broken/summary.json")
    with pytest.raises(ValueError):
        load_separate_gmm_model(tmp_path / "broken")


def test_failed_search_retains_encoders_and_no_assignments(monkeypatch, tmp_path):
    from super_deli.models import separate_gmm

    def failed(*args, **kwargs):
        raise ValueError("deliberate candidate failure")

    monkeypatch.setattr(separate_gmm, "select_gaussian_mixture", failed)
    dataset = write_dataset(data(), tmp_path / "raw")
    with pytest.raises(ValueError, match="deliberate candidate"):
        run_separate_gmm(
            dataset, model_config=config(), base_dir=tmp_path / "raw", output_dir=tmp_path / "fit"
        )
    assert not (tmp_path / "fit/assignments.csv").exists()
    with np.load(tmp_path / "fit/failed-training.npz", allow_pickle=False) as a:
        assert "morphology__scores" in a and "ecology__scores" in a
    assert read_json(tmp_path / "fit/failure.json")["diagnostics"]["failed_role"] == "mixture"


def test_insufficient_component_support_withholds_instead_of_fallback():
    d = data()
    x = np.column_stack((d["morphology"], d["ecology"]))
    with pytest.raises(ValueError, match="No admissible"):
        fit_shared_point(x, config=replace(config(), min_cluster_size=100))
