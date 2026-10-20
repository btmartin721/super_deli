"""Fitted raw covariance, fresh complete refits and failed-attempt preservation."""

# ruff: noqa: E402
import hashlib
import io
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from super_deli.benchmarking.empirical_study import simulate_continuous
from super_deli.io.config import read_json
from super_deli.models.empirical_config import EmpiricalCalibrationConfig
from super_deli.models.empirical_null import RawGaussian, fit_null, simulate_null, write_raw_views
from super_deli.workflows.empirical_calibration import run_empirical_calibration

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/empirical-calibration/calibration.json"
pytestmark = pytest.mark.baseline


def policy():
    value = EmpiricalCalibrationConfig.from_dict(read_json(EXAMPLE))
    return replace(value, n_simulations=2, model=replace(value.model, n_init=1, max_k=2))


def source(tmp_path, case="correlated_gaussian"):
    values, layout, _ = simulate_continuous(case, 20, 193)
    directory = tmp_path / "raw"
    return write_raw_views(values, layout, directory), directory, values, layout


def test_joint_raw_mle_and_complete_refits_are_archived(tmp_path, monkeypatch):
    from super_deli.workflows import integrative

    dataset, directory, values, layout = source(tmp_path)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
    calls = []
    runner = integrative.run_integrative

    def fit(*args, **kwargs):
        calls.append(kwargs["model_config"])
        return runner(*args, **kwargs)

    monkeypatch.setattr(integrative, "run_integrative", fit)
    config = policy()
    output = tmp_path / "fit"
    result = run_empirical_calibration(
        dataset, config=config, base_dir=directory, output_dir=output
    )
    assert len(calls) == 3 and all(
        c.random_seed == config.model.random_seed and c.n_bootstrap == 0 for c in calls
    )
    assert result["successful_fits"] == 3 and result["inference"]["p_value"] >= 1 / 3
    with np.load(output / "null_model.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(values, arrays["values"])
        np.testing.assert_array_equal(values.mean(axis=0), arrays["mean"])
        centered = values - values.mean(axis=0)
        np.testing.assert_array_equal(centered.T @ centered / len(values), arrays["covariance"])
        assert np.max(np.abs(arrays["covariance"][:4, 4:])) > 0.1
    inventory = read_json(output / "archive_inventory.json")
    with ZipFile(output / "pipeline_artifacts.zip") as archive:
        assert set(archive.namelist()) == set(inventory)
        for name, record in inventory.items():
            raw = archive.read(name)
            assert (
                hashlib.sha256(raw).hexdigest() == record["sha256"]
                and len(raw) == record["size_bytes"]
            )
        means = [
            np.load(io.BytesIO(archive.read(prefix + "/run/view_000/parameters/pca.npz")))[
                "feature_means"
            ]
            for prefix in ("observed", "null_0000", "null_0001")
        ]
        assert not np.array_equal(means[0], means[1]) and not np.array_equal(means[1], means[2])
        assert all("truth" not in name for name in archive.namelist())
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    assert not list(output.glob("observed-*")) and not list(output.glob("null-*"))
    with pytest.raises(FileExistsError):
        run_empirical_calibration(dataset, config=config, base_dir=directory, output_dir=output)


@pytest.mark.parametrize("call", [1, 2])
def test_failed_point_and_draw_keep_denominators(tmp_path, monkeypatch, call):
    from super_deli.workflows import integrative

    dataset, directory, *_ = source(tmp_path)
    runner, attempts = integrative.run_integrative, []

    def fail(*args, **kwargs):
        attempts.append(None)
        if len(attempts) == call:
            raise ValueError("injected pipeline failure")
        return runner(*args, **kwargs)

    monkeypatch.setattr(integrative, "run_integrative", fail)
    result = run_empirical_calibration(
        dataset, config=policy(), base_dir=directory, output_dir=tmp_path / "fit"
    )
    assert (
        result["inference"]["p_value"] is None
        and result["inference"]["reject_joint_raw_gaussian"] is None
    )
    assert result["failed_fits"] == 1
    assert result["attempted_fits"] == (1 if call == 1 else 3)
    assert result["unattempted_fits"] == (2 if call == 1 else 0)


@pytest.mark.parametrize("invalid", ["singular", "missing", "count", "coverage", "level"])
def test_unsupported_raw_inputs_fail_before_clustering(tmp_path, invalid):
    from super_deli.core.schema import config_from_dict

    values, layout, _ = simulate_continuous("spherical_gaussian", 20, 193)
    if invalid == "singular":
        values[:, 7] = values[:, 0]
    dataset = write_raw_views(values, layout, tmp_path / "raw")
    document = dataset.to_dict()
    if invalid == "count":
        document["modalities"][0]["features"][0]["dtype"] = "count"
    if invalid == "coverage":
        document["specimens"].append({"specimen_id": "unmeasured"})
    if invalid == "level":
        document["modalities"][0]["level"] = "locality"
        document["localities"] = list(layout["unit_ids"])
    if invalid == "missing":
        p = tmp_path / "raw/view_000.csv"
        text = p.read_text().splitlines()
        fields = text[1].split(",")
        fields[1] = ""
        text[1] = ",".join(fields)
        p.write_text("\n".join(text) + "\n")
    output = tmp_path / "failed"
    with pytest.raises(ValueError):
        run_empirical_calibration(
            config_from_dict(document),
            config=policy(),
            base_dir=tmp_path / "raw",
            output_dir=output,
        )
    failure = read_json(output / "failure.json")
    assert failure["attempted_fits"] == 0 and failure["unattempted_fits"] == 3
    assert not (output / "evidence.json").exists()


def test_null_is_not_an_ordinary_row_bootstrap_and_respects_units(tmp_path):
    from super_deli.io.tabular import ingest_tabular

    dataset, directory, values, _ = source(tmp_path)
    loaded = ingest_tabular(dataset, base_dir=directory, output_dir=tmp_path / "ingested")
    tables = list(loaded.tables)
    second = tables[1]
    tables[1] = replace(
        second,
        values=second.values[::-1],
        unit_ids=second.unit_ids[::-1],
        observation_ids=second.observation_ids[::-1],
    )
    null = fit_null(dataset, tables, policy())
    np.testing.assert_array_equal(null.values, values)
    draw = simulate_null(null, 2026)
    assert not any(np.array_equal(row, candidate) for row in draw for candidate in values)
    scales = np.array([1e-8, 1e8, 1, 1, 1, 1, 1, 1])
    transformed = RawGaussian(
        null.values * scales,
        null.mean * scales,
        null.covariance * scales[:, None] * scales[None, :],
        null.layout,
    )
    np.testing.assert_allclose(
        simulate_null(transformed, 2026), draw * scales, rtol=1e-12, atol=1e-12
    )
