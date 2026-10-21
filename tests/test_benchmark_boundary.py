"""Covariance-cone optimality, dense likelihoods, versioned sampling and failures."""

# ruff: noqa: E402
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")
from scipy.optimize import approx_fprime
from scipy.stats import multivariate_normal

from super_deli.benchmarking.grouped_analysis import point, verify_model
from super_deli.benchmarking.grouped_study import simulate_grouped
from super_deli.io.config import read_json
from super_deli.io.tabular import ingest_tabular
from super_deli.models.grouped_conditional import conditional_likelihood, fit_gaussian, optimality
from super_deli.models.grouped_config import GroupedCalibrationConfig
from super_deli.models.grouped_gaussian import fit_gaussian as legacy_fit
from super_deli.models.grouped_gaussian import sufficient_statistics
from super_deli.models.grouped_null import fit_null, simulate_null, write_raw_views
from super_deli.workflows.grouped_calibration import run_grouped_calibration

pytestmark = pytest.mark.baseline
EXAMPLES = Path(__file__).resolve().parents[1] / "examples/boundary-calibration"


def policy():
    c = GroupedCalibrationConfig.from_dict(read_json(EXAMPLES / "calibration.json"))
    return replace(c, n_simulations=2, model=replace(c.model, n_init=1, max_k=2))


def test_conditional_profile_matches_dense_joint_likelihood_and_derivatives():
    rng = np.random.default_rng(22)
    g, p, q = 9, 2, 2
    counts = np.resize([1, 4, 2, 3], g)
    targets = np.repeat(np.arange(g), counts)
    m, e = rng.normal(size=(len(targets), p)), rng.normal(size=(g, q))
    z, counts, w = sufficient_statistics(m, e, targets)
    ec = e.mean(axis=0)
    ecov = (e - ec).T @ (e - ec) / g
    x = np.column_stack((np.ones(g), e - ec))
    parameters = np.array([0.7, 0.12, 0.5, 0.8, 0.16, 0.6])

    def evaluate(v):
        t = np.array([[v[0], v[1]], [v[1], v[2]]])
        r = np.array([[v[3], v[4]], [v[4], v[5]]])
        return conditional_likelihood(t, r, z[:, :p], x, counts, w, constant=True)

    value, gt, gr, beta = evaluate(parameters)
    numeric = approx_fprime(parameters, lambda v: evaluate(v)[0], 1e-6)
    np.testing.assert_allclose(
        numeric,
        [gt[0, 0], 2 * gt[0, 1], gt[1, 1], gr[0, 0], 2 * gr[0, 1], gr[1, 1]],
        atol=5e-5,
        rtol=2e-4,
    )
    a = beta[1:].T
    t = np.array([[0.7, 0.12], [0.12, 0.5]])
    r = np.array([[0.8, 0.16], [0.16, 0.6]])
    between = np.block([[t + a @ ecov @ a.T, a @ ecov], [ecov @ a.T, ecov]])
    mean = np.r_[beta[0], ec]
    expected = 0.0
    for i, n in enumerate(counts):
        design = np.zeros((n * p + q, p + q))
        design[: n * p, :p] = np.tile(np.eye(p), (n, 1))
        design[n * p :, p:] = np.eye(q)
        covariance = design @ between @ design.T
        covariance[: n * p, : n * p] += np.kron(np.eye(n), r)
        expected -= multivariate_normal.logpdf(
            np.r_[m[targets == i].ravel(), e[i]], mean=design @ mean, cov=covariance
        )
    ecology_nll = -multivariate_normal.logpdf(e, mean=ec, cov=ecov).sum()
    assert value + ecology_nll == pytest.approx(expected, rel=1e-12)


def test_factor_zero_does_not_imply_constrained_stationarity():
    zero = np.zeros((3, 3))
    negative = -np.eye(3)
    assert np.array_equal(2 * negative @ zero, zero)
    assert optimality(zero, negative, zero)["maximum"] == pytest.approx(1)
    assert optimality(zero, np.eye(3), zero)["maximum"] == pytest.approx(0)


@pytest.mark.parametrize(
    "case",
    [
        "independent_gaussian",
        "correlated_gaussian",
        "zero_conditional_gaussian",
        "rank_two_gaussian",
        "weak_conditional_gaussian",
    ],
)
def test_valid_boundary_or_interior_fit_passes_dense_audit_and_regenerates_draws(tmp_path, case):
    values, layout, _ = simulate_grouped(case, 64, 193)
    dataset = write_raw_views(values, layout, tmp_path / "raw")
    c = policy()
    out = tmp_path / "fit"
    result = run_grouped_calibration(dataset, config=c, base_dir=tmp_path / "raw", output_dir=out)
    assert result["successful_fits"] == 3 and result["inference"]["p_value"] is not None
    assert "nonregular" in result["inference"]["p_value_kind"]
    fitted, draw, n, d = verify_model(tmp_path, out, values, layout, c)
    tables = ingest_tabular(
        dataset, base_dir=tmp_path / "raw", output_dir=tmp_path / "ingest"
    ).tables
    null = fit_null(dataset, tables, c)
    assert null.layout["schema_version"] == 2 and null.sampling_factor.shape == (8, 8)
    np.testing.assert_allclose(
        null.sampling_factor @ null.sampling_factor.T, null.covariance, atol=1e-12
    )
    assert null.layout["diagnostics"]["optimality"]["maximum"] <= c.gradient_tol
    with ZipFile(out / "pipeline_artifacts.zip") as archive:
        point(archive, result["point"], c, values, fitted)
        for record in read_json(out / "null_replicates.json"):
            sample = draw(record["simulation_seed"])
            actual = simulate_null(null, record["simulation_seed"])
            for kind in actual:
                np.testing.assert_array_equal(actual[kind], sample[kind])
            point(archive, record, c, sample, fitted)
    if case == "independent_gaussian":
        old = legacy_fit(
            values["morphological"],
            values["ecological"],
            np.repeat(np.arange(64), layout["counts"]),
            replace(c, schema_version=1, solver=None),
        )
        assert (
            fitted["diagnostics"]["negative_log_likelihood"]
            <= old[3]["negative_log_likelihood"] + 1e-6
        )
    if case == "correlated_gaussian":
        assert fitted["diagnostics"]["conditional_rank"] < 4


def test_exact_zero_component_and_sampling_precision(tmp_path):
    values, layout, _ = simulate_grouped("zero_conditional_gaussian", 64, 41)
    targets = np.repeat(np.arange(64), layout["counts"])
    rng = np.random.default_rng(13)
    residual = rng.normal(size=values["morphological"].shape)
    for i in range(64):
        residual[targets == i] -= residual[targets == i].mean(axis=0)
    a = np.eye(4) * 0.6
    values["morphological"] = values["ecological"][targets] @ a.T + residual
    dataset = write_raw_views(values, layout, tmp_path / "raw")
    tables = ingest_tabular(
        dataset, base_dir=tmp_path / "raw", output_dir=tmp_path / "ingest"
    ).tables
    null = fit_null(dataset, tables, policy())
    assert null.layout["diagnostics"]["conditional_rank"] == 0
    np.testing.assert_allclose(null.layout["diagnostics"]["conditional_covariance"], 0, atol=1e-13)
    expected = null.within / layout["counts"][0]
    regression = np.asarray(null.layout["diagnostics"]["regression"])
    draws = []
    for seed in range(800):
        v = simulate_null(null, seed)
        mean = v["morphological"][targets == 0].mean(axis=0)
        draws.append(mean - null.mean[:4] - regression @ (v["ecological"][0] - null.mean[4:]))
    scale = np.sqrt(np.diag(expected))
    cov = np.cov(draws, rowvar=False)
    np.testing.assert_allclose(
        cov / scale[:, None] / scale[None, :],
        expected / scale[:, None] / scale[None, :],
        atol=0.12,
        rtol=0.12,
    )


def test_unresolved_optimization_and_archive_failures_keep_denominators(tmp_path):
    values, layout, _ = simulate_grouped("complementary", 64, 193)
    dataset = write_raw_views(values, layout, tmp_path / "raw")
    c = policy()
    c = replace(c, max_iterations=1, solver=replace(c.solver, max_polish_iterations=1))
    with pytest.raises(ValueError, match="no constrained converged fit"):
        run_grouped_calibration(
            dataset, config=c, base_dir=tmp_path / "raw", output_dir=tmp_path / "failed"
        )
    failure = read_json(tmp_path / "failed/failure.json")
    assert failure["attempted_fits"] == 0 and failure["unattempted_fits"] == 3
    with pytest.raises(ValueError, match="max_archive_bytes"):
        run_grouped_calibration(
            dataset,
            config=replace(policy(), max_archive_bytes=1),
            base_dir=tmp_path / "raw",
            output_dir=tmp_path / "archive",
        )
    assert read_json(tmp_path / "archive/failure.json")["attempted_fits"] == 1


def test_units_and_ecology_row_order_do_not_change_the_fitted_law(tmp_path):
    values, layout, _ = simulate_grouped("correlated_gaussian", 64, 193)
    dataset = write_raw_views(values, layout, tmp_path / "raw")
    tables = ingest_tabular(
        dataset, base_dir=tmp_path / "raw", output_dir=tmp_path / "ingest"
    ).tables
    original = fit_null(dataset, tables, policy())
    scale = np.array([1e-6, 1e3, 0.2, 7.0])
    changed = tuple(
        replace(
            t,
            values=tuple(
                tuple(float(float(x) * s) for x, s in zip(row, scale, strict=True))
                for row in t.values
            ),
        )
        for t in tables
    )
    scaled = fit_null(dataset, changed, policy())
    s = np.tile(scale, 2)
    np.testing.assert_allclose(scaled.mean / s, original.mean, atol=3e-6)
    np.testing.assert_allclose(
        scaled.covariance / s[:, None] / s[None, :], original.covariance, atol=3e-6
    )
    reversed_tables = tuple(
        replace(
            t,
            values=t.values[::-1],
            unit_ids=t.unit_ids[::-1],
            observation_ids=t.observation_ids[::-1],
        )
        if t.kind == "ecological"
        else t
        for t in tables
    )
    reverse = fit_null(dataset, reversed_tables, policy())
    np.testing.assert_allclose(reverse.covariance, original.covariance, atol=1e-10)


@pytest.mark.parametrize(
    "case,repetition", [("correlated_gaussian", 23), ("complementary", 20), ("heavy_tailed", 11)]
)
def test_previous_iteration_limit_datasets_pass_constrained_checks(case, repetition):
    from super_deli.benchmarking.config import derived_seed

    seed = derived_seed(20261024, case, repetition, "observed_data")
    values, layout, _ = simulate_grouped(case, 64, seed)
    *_, diagnostics = fit_gaussian(
        values["morphological"],
        values["ecological"],
        np.repeat(np.arange(64), layout["counts"]),
        policy(),
    )
    assert diagnostics["optimality"]["maximum"] <= policy().gradient_tol
    assert any(a["accepted"] for a in diagnostics["attempts"])


def test_conditional_gls_budget_is_checked_before_large_coefficient_solve():
    rng = np.random.default_rng(10)
    c = policy()
    c = replace(c, model=replace(c.model, max_working_bytes=5_000_000))
    with pytest.raises(ValueError, match="GLS workspace"):
        fit_gaussian(
            rng.normal(size=(88, 20)), rng.normal(size=(44, 20)), np.repeat(np.arange(44), 2), c
        )
