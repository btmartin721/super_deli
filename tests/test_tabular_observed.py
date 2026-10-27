# ruff: noqa: E402
"""Independent marginal, moment, constraint and failure audits."""

import json
from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from super_deli.models.observed_config import ObservedGMMConfig
from super_deli.models.observed_gmm import fit_observed_matrix, fit_start
from super_deli.models.observed_likelihood import (
    Parameters,
    conditional_offsets,
    constrain_covariance,
    evaluate,
    patterns,
    update,
)

# Optional extras must not be imported by foundation-only tests.


def parameters():
    return Parameters(
        np.array([0.4, 0.6]),
        np.array([[0.0, 1.0], [2.0, -1.0]]),
        np.array([[[2.0, 0.7], [0.7, 1.0]], [[1.0, -0.2], [-0.2, 3.0]]]),
    )


def test_marginals_and_conditional_second_moments():
    p = parameters()
    x = np.array([[1.0, np.nan], [np.nan, 2.0], [0.1, 0.2]])
    density, membership = evaluate(len(x), p, patterns(x))
    joint = np.empty((len(x), 2))
    for i, row in enumerate(x):
        o = np.flatnonzero(~np.isnan(row))
        for k in range(2):
            joint[i, k] = p.weights[k] * multivariate_normal.pdf(
                row[o], p.means[k, o], p.covariances[k][np.ix_(o, o)]
            )
    np.testing.assert_allclose(np.exp(density), joint.sum(axis=1))
    np.testing.assert_allclose(membership, joint / joint.sum(axis=1)[:, None])
    g = patterns(x[:1])[0]
    off, cov = conditional_offsets(g, p.means[0], p.covariances[0])
    assert off[0, 1] == pytest.approx(0.7 / 2)
    assert cov[1, 1] == pytest.approx(1 - 0.7**2 / 2)
    # A conditional-mean-only update would incorrectly omit this nonzero variance.
    draws = np.random.default_rng(902).normal(off[0, 1], np.sqrt(cov[1, 1]), 100000)
    assert draws.var() == pytest.approx(cov[1, 1], rel=0.02)


@pytest.mark.parametrize("s", [np.diag([0.0001, 2.0]), np.array([[2.0, 0.7], [0.7, 1.0]])])
def test_constrained_covariance_matches_independent_optimizer(s):
    bound = 0.2
    fitted = constrain_covariance(s, bound)

    def objective(v):
        c = np.array([[v[0], v[1]], [v[1], v[2]]])
        sign, logdet = np.linalg.slogdet(c)
        return 1e10 if sign <= 0 else logdet + np.trace(np.linalg.solve(c, s))

    def constraint(v):
        return np.linalg.eigvalsh([[v[0], v[1]], [v[1], v[2]]]) - bound

    result = minimize(
        objective,
        [1.0, 0.0, 2.0],
        method="SLSQP",
        constraints={"type": "ineq", "fun": constraint},
        options={"ftol": 1e-11, "maxiter": 1000},
    )
    assert result.success
    assert objective([fitted[0, 0], fitted[0, 1], fitted[1, 1]]) == pytest.approx(
        result.fun, abs=1e-7
    )


def test_complete_data_update_and_k1_solution():
    x = np.random.default_rng(903).normal(size=(40, 2))
    p = parameters()
    _, r = evaluate(len(x), p, patterns(x))
    got = update(len(x), p, patterns(x), 1e-4)
    for k in range(2):
        mu = r[:, k] @ x / r[:, k].sum()
        cov = ((x - mu).T * r[:, k]) @ (x - mu) / r[:, k].sum()
        np.testing.assert_allclose(got.means[k], mu)
        np.testing.assert_allclose(got.covariances[k], constrain_covariance(cov, 1e-4))
    fit = fit_observed_matrix(x, config=ObservedGMMConfig(("a", "b"), max_k=1, n_init=1))
    assert fit.summary["status"] == "complete"
    np.testing.assert_allclose(fit.normalization.means, x.mean(axis=0))
    np.testing.assert_allclose(fit.parameters.means, 0, atol=1e-12)
    np.testing.assert_allclose(
        fit.parameters.covariances[0],
        np.cov(fit.normalization.transform(x), rowvar=False, bias=True),
        atol=1e-12,
    )


def test_ascent_mask_preservation_projection_and_reproducibility():
    x = np.random.default_rng(904).normal(size=(48, 2))
    x[0, :] = np.nan
    x[1:10, 0] = np.nan
    original = x.copy()
    config = ObservedGMMConfig(("a", "b"), max_k=2, n_init=2)
    a = fit_observed_matrix(x, config=config)
    b = fit_observed_matrix(x, config=config)
    assert a.summary == b.summary
    np.testing.assert_array_equal(x, original)
    assert a.summary["excluded_all_missing"] == 1
    for key, value in a.arrays.items():
        np.testing.assert_array_equal(value, b.arrays[key])
        if key.endswith("_trace"):
            assert np.diff(value).min(initial=0) >= -1e-8
    density, r = a.predict(np.array([[0.0, np.nan], [np.nan, np.nan], [100.0, 101.0]]))
    assert np.isnan(density[1]) and np.isnan(r[1]).all()
    assert np.isfinite(density[[0, 2]]).all()
    expected = (
        sum(
            a.parameters.weights[k]
            * multivariate_normal.pdf(
                (0 - a.normalization.means[0]) / a.normalization.scales[0],
                a.parameters.means[k, 0],
                a.parameters.covariances[k, 0, 0],
            )
            for k in range(a.summary["selected_k"])
        )
        / a.normalization.scales[0]
    )
    assert density[0] == pytest.approx(np.log(expected))


@pytest.mark.parametrize("case", ["pair", "constant", "unobserved", "workspace", "infinite"])
def test_information_and_memory_guards(case):
    x = np.random.default_rng(905).normal(size=(32, 2))
    config = ObservedGMMConfig(("a", "b"))
    if case == "pair":
        x[:16, 0] = np.nan
        x[16:, 1] = np.nan
    elif case == "constant":
        x[:, 0] = 1
    elif case == "unobserved":
        x[:, 0] = np.nan
    elif case == "workspace":
        config = replace(config, max_working_bytes=1)
    else:
        x[0, 0] = np.inf
    with pytest.raises((ValueError, FloatingPointError)):
        fit_observed_matrix(x, config=config)


def test_unconverged_candidate_withholds_selection():
    x = np.random.default_rng(906).normal(size=(32, 2))
    x[:8, 0] = np.nan
    f = fit_observed_matrix(x, config=ObservedGMMConfig(("a", "b"), max_k=2, n_init=1, max_iter=1))
    assert f.summary["status"] == "failed" and f.summary["selected_k"] is None
    assert len(f.summary["candidates"]) == 2
    assert any(c["status"] == "numerical_failure" for c in f.summary["candidates"])
    with pytest.raises(ValueError):
        f.predict(x)


def test_workflow_raw_snapshot_and_saved_prediction(tmp_path):
    from pathlib import Path

    from super_deli.core.schema import config_from_dict
    from super_deli.workflows.observed_gmm import load_observed_model, run_observed_gmm

    base = Path(__file__).parents[1] / "examples/tabular-missing/fixture"
    dataset = config_from_dict(json.loads((base / "dataset.json").read_text()))
    config = ObservedGMMConfig(tuple(f"trait_{i}" for i in range(4)), max_k=1, n_init=1)
    root = tmp_path / "run"
    summary = run_observed_gmm(dataset, model_config=config, base_dir=base, output_dir=root)
    assert summary["status"] == "complete"
    assert summary["p_value"] is None and summary["excluded_all_missing"] == 1
    assert (root / "input.tsv").read_bytes() == (base / "traits.tsv").read_bytes()
    loaded = load_observed_model(root)
    assert np.isfinite(loaded.predict(np.zeros((2, 4)))[0]).all()
    assert "excluded_all_missing" in (root / "membership.csv").read_text()
    with (root / "model.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError):
        load_observed_model(root)


def test_complete_data_em_matches_independent_updates_and_sklearn():
    from scipy.special import logsumexp
    from sklearn.mixture import GaussianMixture

    x = np.random.default_rng(908).normal(size=(160, 2))
    x[:80] += 4
    initial = Parameters(
        np.array([0.5, 0.5]),
        np.array([[0.0, 0.0], [4.0, 4.0]]),
        np.repeat(np.eye(2)[None], 2, axis=0),
    )
    config = ObservedGMMConfig(("a", "b"), max_k=2, n_init=1, tol=1e-10, fixed_point_tol=1e-7)
    record, evidence = fit_start(x, initial, patterns(x), config)
    assert record["status"] == "converged"
    state = initial
    for _ in range(record["iterations"]):
        log = np.column_stack(
            [
                np.log(state.weights[k])
                + multivariate_normal.logpdf(x, state.means[k], state.covariances[k])
                for k in range(2)
            ]
        )
        r = np.exp(log - logsumexp(log, axis=1)[:, None])
        mass = r.sum(axis=0)
        means = (r.T @ x) / mass[:, None]
        cov = np.array(
            [
                constrain_covariance(
                    ((x - means[k]).T * r[:, k]) @ (x - means[k]) / mass[k], config.covariance_bound
                )
                for k in range(2)
            ]
        )
        state = Parameters(mass / len(x), means, cov)
    for key in ("weights", "means", "covariances"):
        np.testing.assert_allclose(getattr(evidence["final"], key), getattr(state, key), atol=1e-10)
    sklearn = GaussianMixture(
        n_components=2,
        weights_init=initial.weights,
        means_init=initial.means,
        precisions_init=np.linalg.inv(initial.covariances),
        reg_covar=0,
        tol=1e-10,
        max_iter=2000,
        random_state=1,
    ).fit(x)
    assert np.linalg.eigvalsh(state.covariances).min() > config.covariance_bound
    assert record["log_likelihood"] == pytest.approx(sklearn.score(x) * len(x), abs=1e-6)


def test_independent_incomplete_update_and_freeze_budgets():
    from super_deli.benchmarking.observed_audit import reference, reference_update
    from super_deli.benchmarking.observed_study import SETTINGS, tasks

    x = np.random.default_rng(909).normal(size=(50, 2))
    x[:20, 0] = np.nan
    p = parameters()
    dense = reference_update(x, (p.weights, p.means, p.covariances), 0.0001)
    got = update(len(x), p, patterns(x), 0.0001)
    for key, expected in zip(("weights", "means", "covariances"), dense, strict=True):
        np.testing.assert_allclose(getattr(got, key), expected, atol=1e-12)
    np.testing.assert_allclose(
        reference(x, p.weights, p.means, p.covariances)[0], evaluate(len(x), p, patterns(x))[0]
    )
    assert len(tasks("main")) == 60 and len(tasks("preflight")) == 6
    assert len(tasks("main")) * 20 + 24 == SETTINGS["main_searches"] == 1224
    assert len(tasks("preflight")) * 24 == SETTINGS["preflight_searches"] == 144


def test_cli_observed_opt_in(tmp_path, capsys):
    from pathlib import Path

    from super_deli.cli import main

    base = Path(__file__).parents[1] / "examples/tabular-observed"
    assert (
        main(
            [
                "cluster-tabular-observed",
                str(base / "dataset.json"),
                "--model-config",
                str(base / "model.json"),
                "--output-dir",
                str(tmp_path / "cli"),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["p_value"] is None and summary["stability"] == "not_computed"


def test_complete_point_evidence_passes_independent_audit(tmp_path):
    from super_deli.benchmarking.observed_audit import audit_fit
    from super_deli.benchmarking.observed_study import NAMES, save_fit
    from super_deli.models.observed_gmm import normalization

    rng = np.random.default_rng(912)
    x = rng.normal(size=(40, 4))
    masked = x.copy()
    masked[rng.random(x.shape) < 0.2] = np.nan
    norm = normalization(masked)
    evaluation = rng.normal(size=(30, 4))
    mask = rng.random(evaluation.shape) < 0.3
    config = ObservedGMMConfig(NAMES)
    folder = tmp_path / "point"
    save_fit(
        masked,
        norm,
        config,
        folder,
        dict(arm="observed"),
        np.arange(len(x)),
        evaluation,
        mask,
        np.zeros(len(x)),
    )
    result = audit_fit(
        folder,
        masked,
        np.arange(len(x)),
        norm.means,
        norm.scales,
        evaluation,
        mask,
        np.zeros(len(x)),
    )
    assert result["starts"] == 40


def test_write_once_reporting_and_exact_resume(tmp_path):
    from super_deli.benchmarking.observed_study import NAMES, save_fit, write_once_json
    from super_deli.models.observed_gmm import normalization

    x = np.random.default_rng(913).normal(size=(32, 4))
    norm = normalization(x)
    configuration = ObservedGMMConfig(NAMES, max_k=1, n_init=1)
    arguments = (
        x,
        norm,
        configuration,
        tmp_path / "fit",
        dict(arm="unit_test"),
        np.arange(len(x)),
        x,
        np.zeros_like(x, dtype=bool),
        np.zeros(len(x)),
    )
    result = save_fit(*arguments)
    original = {p.name: p.read_bytes() for p in (tmp_path / "fit").iterdir()}
    assert save_fit(*arguments, resume=True) == result
    assert original == {p.name: p.read_bytes() for p in (tmp_path / "fit").iterdir()}
    with pytest.raises(FileExistsError):
        save_fit(*arguments)
    sidecar = tmp_path / "comparison.json"
    write_once_json(dict(reference_common_n=32), sidecar, False)
    write_once_json(dict(reference_common_n=32), sidecar, True)
    with pytest.raises(ValueError):
        write_once_json(dict(reference_common_n=31), sidecar, True)


@pytest.mark.parametrize("case", ["spherical", "correlated", "four_groups"])
def test_independent_source_reconstruction_on_invented_nonstudy_inputs(case):
    from super_deli.benchmarking.observed_audit import source_arrays
    from super_deli.benchmarking.observed_study import masks, sample

    # This namespace is outside the frozen preflight/main grid.
    task = dict(case=case, n=32, replicate=999, phase="main")
    namespace = [1, ["spherical", "correlated", "four_groups"].index(case), 32, 999]
    expected = source_arrays(task)
    x, truth = sample(case, 32, [80403] + namespace)
    np.testing.assert_array_equal(expected["complete"], x)
    np.testing.assert_array_equal(expected["truth"], truth)
    for name, mask in masks(x, [80409] + namespace).items():
        np.testing.assert_array_equal(expected[f"mask_{name}"], mask)
    evaluation, labels = sample(case, 2048, [80429] + namespace + [0])
    np.testing.assert_array_equal(expected["evaluation"], evaluation)
    np.testing.assert_array_equal(expected["evaluation_truth"], labels)
    for name, mask in masks(evaluation, [80429] + namespace + [1]).items():
        np.testing.assert_array_equal(expected[f"eval_mask_{name}"], mask)
