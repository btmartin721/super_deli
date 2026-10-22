"""Dense raw likelihood, derivatives, fixed-design refits and failure accounting."""

# ruff: noqa: E402
import io
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")
from scipy.optimize import approx_fprime
from scipy.stats import multivariate_normal

from super_deli.benchmarking.grouped_study import simulate_grouped
from super_deli.core.schema import config_from_dict
from super_deli.io.config import read_json
from super_deli.models.grouped_config import GroupedCalibrationConfig
from super_deli.models.grouped_gaussian import (
    pack_covariance,
    profile_likelihood,
    sufficient_statistics,
)
from super_deli.models.grouped_null import fit_null, simulate_null, write_raw_views
from super_deli.workflows.grouped_calibration import run_grouped_calibration

pytestmark = pytest.mark.baseline
EXAMPLE = Path(__file__).resolve().parents[1] / "examples/grouped-calibration/calibration.json"


def policy():
    c = GroupedCalibrationConfig.from_dict(read_json(EXAMPLE))
    return replace(c, n_simulations=2, model=replace(c.model, n_init=1, max_k=2))


def source(tmp_path, level="locality"):
    values, layout, _ = simulate_grouped("independent_gaussian", 64, 193, target_level=level)
    dataset = write_raw_views(values, layout, tmp_path / "raw")
    return dataset, values, layout


def test_profile_likelihood_matches_dense_individual_covariance_and_derivatives():
    rng = np.random.default_rng(4)
    counts = np.array([1, 4, 2, 3, 1])
    targets = np.repeat(np.arange(5), counts)
    morphology, ecology = rng.normal(size=(len(targets), 2)), rng.normal(size=(5, 2))
    a = rng.normal(size=(4, 4))
    between = a @ a.T + np.eye(4)
    a = rng.normal(size=(2, 2))
    within = a @ a.T + np.eye(2)
    parameters = np.r_[pack_covariance(between), pack_covariance(within)]
    z, counts, scatter = sufficient_statistics(morphology, ecology, targets)
    value, gradient, mean = profile_likelihood(parameters, z, counts, scatter, constant=True)
    expected = 0.0
    for g, n in enumerate(counts):
        design = np.zeros((2 * n + 2, 4))
        design[: 2 * n, :2] = np.tile(np.eye(2), (n, 1))
        design[2 * n :, 2:] = np.eye(2)
        covariance = design @ between @ design.T
        covariance[: 2 * n, : 2 * n] += np.kron(np.eye(n), within)
        expected -= multivariate_normal.logpdf(
            np.r_[morphology[targets == g].ravel(), ecology[g]], mean=design @ mean, cov=covariance
        )
    assert value == pytest.approx(expected, rel=1e-12)
    numeric = approx_fprime(
        parameters, lambda v: profile_likelihood(v, z, counts, scatter, constant=True)[0], 1e-6
    )
    np.testing.assert_allclose(gradient, numeric, atol=3e-5, rtol=1e-4)


@pytest.mark.parametrize("level", ["locality", "population"])
def test_unequal_counts_raw_refits_and_independent_dense_audit(tmp_path, monkeypatch, level):
    from super_deli.benchmarking.grouped_analysis import point, verify_model
    from super_deli.workflows import integrative

    dataset, values, layout = source(tmp_path, level)
    c = policy()
    c = replace(c, model=replace(c.model, target_level=level))
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (tmp_path / "raw").iterdir()}
    runner, calls = integrative.run_integrative, []

    def counted(*args, **kwargs):
        calls.append(kwargs["model_config"])
        return runner(*args, **kwargs)

    monkeypatch.setattr(integrative, "run_integrative", counted)
    out = tmp_path / "fit"
    result = run_grouped_calibration(dataset, config=c, base_dir=tmp_path / "raw", output_dir=out)
    assert result["successful_fits"] == 3 and len(calls) == 3
    assert all(v.random_seed == c.model.random_seed and v.n_bootstrap == 0 for v in calls)
    fitted, draw, n, d = verify_model(tmp_path, out, values, layout, c)
    assert (n, d) == (64, 8)
    assert sorted(set(fitted["counts"])) == [1, 2, 3, 4]
    with ZipFile(out / "pipeline_artifacts.zip") as archive:
        point(archive, result["point"], c, values, fitted)
        for record in read_json(out / "null_replicates.json"):
            sample = draw(record["simulation_seed"])
            point(archive, record, c, sample, fitted)
            declaration = read_json(out / "configuration.json")
            assert declaration["model"]["target_level"] == level
            assert not any("truth" in name for name in archive.namelist())
        means = [
            np.load(io.BytesIO(archive.read(prefix + "/run/view_000/parameters/pca.npz")))[
                "feature_means"
            ]
            for prefix in ("observed", "null_0000", "null_0001")
        ]
        assert not np.array_equal(means[0], means[1]) and not np.array_equal(means[1], means[2])
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    with pytest.raises(FileExistsError):
        run_grouped_calibration(dataset, config=c, base_dir=tmp_path / "raw", output_dir=out)


@pytest.mark.parametrize("call", [1, 2])
def test_failed_point_or_draw_withholds_inference(tmp_path, monkeypatch, call):
    from super_deli.workflows import integrative

    dataset, _, _ = source(tmp_path)
    runner, calls = integrative.run_integrative, []

    def failed(*args, **kwargs):
        calls.append(None)
        if len(calls) == call:
            raise ValueError("injected failure")
        return runner(*args, **kwargs)

    monkeypatch.setattr(integrative, "run_integrative", failed)
    result = run_grouped_calibration(
        dataset, config=policy(), base_dir=tmp_path / "raw", output_dir=tmp_path / "fit"
    )
    assert result["inference"]["p_value"] is None
    assert result["inference"]["reject_grouped_raw_gaussian"] is None
    assert result["failed_fits"] == 1
    assert result["attempted_fits"] == (1 if call == 1 else 3)
    assert result["unattempted_fits"] == (2 if call == 1 else 0)


@pytest.mark.parametrize(
    "invalid",
    ["singular_within", "count", "coverage", "no_replication", "nonconvergence", "archive"],
)
def test_invalid_null_or_archive_keeps_unattempted_counts(tmp_path, invalid):
    dataset, values, layout = source(tmp_path)
    c = policy()
    document = dataset.to_dict()
    if invalid == "count":
        document["modalities"][0]["features"][0]["dtype"] = "count"
    if invalid == "coverage":
        document["localities"].append("unobserved")
    if invalid in {"singular_within", "no_replication"}:
        if invalid == "singular_within":
            values["morphological"][:, 3] = values["morphological"][:, 0]
        else:
            keep = np.cumsum([0, *layout["counts"][:-1]])
            values["morphological"] = values["morphological"][keep]
            layout["specimens"] = [layout["specimens"][i] for i in keep]
            for key in ("unit_ids", "observation_ids", "target_ids"):
                layout["views"][0][key] = [layout["views"][0][key][i] for i in keep]
        dataset = write_raw_views(values, layout, tmp_path / "invalid")
        directory = tmp_path / "invalid"
    else:
        dataset = config_from_dict(document)
        directory = tmp_path / "raw"
    if invalid == "nonconvergence":
        c = replace(c, max_iterations=1)
    if invalid == "archive":
        c = replace(c, max_archive_bytes=1)
    out = tmp_path / "fit"
    with pytest.raises(ValueError):
        run_grouped_calibration(dataset, config=c, base_dir=directory, output_dir=out)
    failure = read_json(out / "failure.json")
    assert failure["attempted_fits"] == (1 if invalid == "archive" else 0)
    assert failure["unattempted_fits"] == (2 if invalid == "archive" else 3)
    assert not (out / "evidence.json").exists()


def test_identity_order_feature_units_and_scale_equivariance(tmp_path):
    from super_deli.io.tabular import ingest_tabular

    dataset, _, _ = source(tmp_path)
    tables = ingest_tabular(
        dataset, base_dir=tmp_path / "raw", output_dir=tmp_path / "ingestion"
    ).tables
    c = policy()
    original = fit_null(dataset, tables, c)
    reversed_tables = tuple(
        replace(
            t,
            values=t.values[::-1],
            observation_ids=t.observation_ids[::-1],
            unit_ids=t.unit_ids[::-1],
        )
        if t.kind == "ecological"
        else t
        for t in tables
    )
    reordered = fit_null(dataset, reversed_tables, c)
    np.testing.assert_allclose(original.mean, reordered.mean, atol=1e-10)
    np.testing.assert_allclose(original.covariance, reordered.covariance, atol=1e-9)
    for kind, expected in simulate_null(original, 10).items():
        sample = simulate_null(reordered, 10)[kind]
        np.testing.assert_array_equal(sample[::-1] if kind == "ecological" else sample, expected)
    scaling = np.array([1e-3, 1e3, 0.2, 7.0])
    transformed = tuple(
        replace(
            t,
            values=tuple(
                tuple(float(float(x) * s) for x, s in zip(row, scaling, strict=True))
                for row in t.values
            ),
        )
        for t in tables
    )
    scaled = fit_null(dataset, transformed, c)
    scale = np.tile(scaling, 2)
    np.testing.assert_allclose(scaled.mean / scale, original.mean, atol=1e-6)
    np.testing.assert_allclose(
        scaled.covariance / scale[:, None] / scale[None, :], original.covariance, atol=3e-6
    )
