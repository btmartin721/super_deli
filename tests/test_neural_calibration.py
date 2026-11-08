"""Raw null, full-refit parity, failed draw withholding and identity provenance."""

# ruff: noqa: E402

from dataclasses import replace
from io import BytesIO
from zipfile import ZipFile

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("sklearn")

from super_deli.benchmarking.matched_data import write_dataset
from super_deli.io.config import read_json
from super_deli.models.neural_calibration_config import NeuralCalibrationConfig
from super_deli.models.separate_gmm_config import SeparateGMMConfig
from super_deli.workflows.neural_calibration import fit_input, run_neural_calibration
from super_deli.workflows.separate_gmm import run_separate_gmm


def setup(tmp_path):
    rng = np.random.default_rng(61919)
    data = dict(
        morphology=rng.normal(size=(32, 4)),
        ecology=rng.normal(size=(32, 8)),
        ecology_order=rng.permutation(32),
    )
    ds = write_dataset(data, tmp_path / "raw")
    from super_deli.benchmarking.matched_data import configuration
    from super_deli.models.separate_gmm_config import FAMILY

    model = SeparateGMMConfig.from_dict(
        {**configuration("paired_0", 0).to_dict(), "model_family": FAMILY}
    )
    return data, ds, NeuralCalibrationConfig(True, model, n_simulations=2)


def test_compact_point_is_exact_native_point(tmp_path):
    _, ds, c = setup(tmp_path)
    a = fit_input(ds, model_config=c.model, base_dir=tmp_path / "raw", output_dir=tmp_path / "a")
    b = run_separate_gmm(
        ds, model_config=c.model, base_dir=tmp_path / "raw", output_dir=tmp_path / "b"
    )
    assert a["selected_k"] == b["selected_k"]
    with np.load(tmp_path / "a/point.npz") as x, np.load(tmp_path / "b/point.npz") as y:
        assert set(x.files) == set(y.files)
        for name in x.files:
            np.testing.assert_array_equal(x[name], y[name])


def test_raw_null_and_all_draws_refit(tmp_path):
    data, ds, c = setup(tmp_path)
    out = tmp_path / "calibration"
    result = run_neural_calibration(ds, config=c, base_dir=tmp_path / "raw", output_dir=out)
    assert result["attempted_fits"] == result["successful_fits"] == 3
    assert result["inference"]["p_value"] in (1 / 3, 2 / 3, 1)
    assert result["inference"]["requested"] == 2
    x = np.column_stack((data["morphology"], data["ecology"]))
    with np.load(out / "null_model.npz") as n:
        np.testing.assert_array_equal(n["values"], x)
        np.testing.assert_allclose(n["mean"], x.mean(0), atol=1e-15)
        np.testing.assert_allclose(n["covariance"], np.cov(x, rowvar=False, bias=True), atol=1e-15)
    with ZipFile(out / "pipeline_artifacts.zip") as z:
        fits = []
        for prefix in ("observed", "null_0000", "null_0001"):
            with np.load(BytesIO(z.read(prefix + "/run/point.npz"))) as a:
                fits.append(a["morphology__means"].copy())
        assert not np.array_equal(fits[0], fits[1])
        assert not np.array_equal(fits[1], fits[2])
        assert set(z.namelist()) == set(read_json(out / "archive_inventory.json"))
    with pytest.raises(FileExistsError):
        run_neural_calibration(ds, config=c, base_dir=tmp_path / "raw", output_dir=out)


def test_failed_requested_draw_withholds_pvalue(monkeypatch, tmp_path):
    from super_deli.workflows import neural_calibration as module

    _, ds, c = setup(tmp_path)
    original, calls = module.fit_input, []

    def fit(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("Deliberate numerical draw failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "fit_input", fit)
    result = run_neural_calibration(
        ds, config=c, base_dir=tmp_path / "raw", output_dir=tmp_path / "calibration"
    )
    assert len(calls) == 3 and result["failed_fits"] == 1
    assert result["inference"]["p_value"] is None
    assert result["inference"]["reject_joint_raw_gaussian"] is None
    assert result["inference"]["failed"] == 1


def test_archive_limit_failure_is_retained(tmp_path):
    _, ds, c = setup(tmp_path)
    with pytest.raises(ValueError, match="archive"):
        run_neural_calibration(
            ds,
            config=replace(c, max_archive_bytes=1),
            base_dir=tmp_path / "raw",
            output_dir=tmp_path / "calibration",
        )
    assert read_json(tmp_path / "calibration/failure.json")["status"] == "failed"


def test_repeated_units_rejected_before_null_estimation(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from super_deli.benchmarking.matched_audit import synthetic_tables
    from super_deli.io import tabular
    from super_deli.workflows.neural_calibration import load_raw_null

    data, ds, c = setup(tmp_path)
    tables = list(synthetic_tables(data))
    ids = list(tables[0].unit_ids)
    ids[1] = ids[0]
    tables[0] = replace(tables[0], unit_ids=tuple(ids))
    monkeypatch.setattr(tabular, "ingest_tabular", lambda *a, **kw: SimpleNamespace(tables=tables))
    calls = []
    with pytest.raises(ValueError, match="unique"):
        load_raw_null(ds, c, tmp_path / "raw", tmp_path / "calibration", lambda *a: calls.append(1))
    assert calls == []
