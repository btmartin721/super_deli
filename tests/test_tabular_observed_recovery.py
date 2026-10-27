# ruff: noqa: E402
"""Continuation accounting, conditional likelihood and diagnostic isolation."""

from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from super_deli.benchmarking.observed_recovery import (
    assemble,
    continued_start,
    get_parameters,
    oracle_initial,
    put_parameters,
)
from super_deli.benchmarking.observed_recovery_audit import independent_inputs
from super_deli.benchmarking.observed_recovery_study import SETTINGS, make_inputs, tasks
from super_deli.models.observed_config import ObservedGMMConfig
from super_deli.models.observed_gmm import fit_observed_matrix, fit_start, initialize
from super_deli.models.observed_likelihood import patterns


def path_fixture():
    x = np.random.default_rng(92001).normal(size=(40, 2))
    x[:10, 0] = np.nan
    config = ObservedGMMConfig(("a", "b"), max_iter=3, tol=1e-20, fixed_point_tol=1e-20)
    initial = initialize(x, 2, 18, config.covariance_bound)
    record, evidence = fit_start(x, initial, patterns(x), config)
    record.update(start=0, seed=18)
    a = {"k2_s0_trace": evidence["trace"]}
    for stage in ("initial", "previous", "final"):
        put_parameters(a, "k2_s0_" + stage + "_", evidence[stage])
    return x, config, initial, record, a


def test_continuation_matches_uninterrupted_updates_and_preserves_original():
    x, config, initial, record, a = path_fixture()
    original = {key: value.copy() for key, value in a.items()}
    changed, b = continued_start(x, a, "k2_s0_", record, config, total_cap=9)
    uninterrupted, evidence = fit_start(x, initial, patterns(x), replace(config, max_iter=9))
    assert record["iterations"] == 3 and changed["iterations"] == 9
    assert changed["additional_iterations"] == 6
    assert changed["status"] == uninterrupted["status"]
    np.testing.assert_array_equal(b["k2_s0_trace"], evidence["trace"])
    for name in ("weights", "means", "covariances"):
        np.testing.assert_array_equal(b["k2_s0_final_" + name], getattr(evidence["final"], name))
    for key in a:
        np.testing.assert_array_equal(a[key], original[key])


@pytest.mark.parametrize("problem", ["objective", "trace", "status", "iterations", "cap"])
def test_continuation_rejects_wrong_anchors_and_nonexhausted_starts(problem):
    x, config, _, record, a = path_fixture()
    cap = 9
    if problem == "objective":
        record["log_likelihood"] += 1
    elif problem == "trace":
        a["k2_s0_trace"] = a["k2_s0_trace"][:-1]
    elif problem == "status":
        record["status"] = "converged"
    elif problem == "iterations":
        record["iterations"] -= 1
    else:
        cap = config.max_iter
    with pytest.raises(ValueError):
        continued_start(x, a, "k2_s0_", record, config, cap)


def test_no_continuation_reconstruction_matches_production_and_ignores_truth():
    x = np.random.default_rng(92002).normal(size=(64, 2))
    config = ObservedGMMConfig(("a", "b"), max_k=2, n_init=2)
    fitted = fit_observed_matrix(x, config=config)
    a = {key: value.copy() for key, value in fitted.arrays.items()}
    candidates, selected = assemble(fitted.summary["candidates"], a, config)
    for actual, expected in zip(candidates, fitted.summary["candidates"], strict=True):
        assert actual == expected
    assert len(selected.weights) == fitted.summary["selected_k"]
    for name in ("weights", "means", "covariances", "labels", "membership"):
        np.testing.assert_array_equal(a[name], fitted.arrays[name])


def test_failed_feasible_candidate_withholds_even_when_k1_is_valid():
    x = np.random.default_rng(92003).normal(size=(32, 2))
    config = ObservedGMMConfig(("a", "b"), max_k=2, n_init=1)
    fitted = fit_observed_matrix(x, config=config)
    candidates = [dict(c) for c in fitted.summary["candidates"]]
    candidates[1]["starts"] = [
        dict(start=0, seed=1, status="failed", error="Unresolved", iterations=0)
    ]
    rebuilt, selected = assemble(candidates, fitted.arrays.copy(), config)
    assert selected is None and rebuilt[1]["status"] == "numerical_failure"


@pytest.mark.parametrize("phase", ["preflight", "main"])
@pytest.mark.parametrize("case", ["spherical", "correlated", "four_groups"])
def test_independent_law_and_mask_reconstruction_outside_scientific_grid(phase, case):
    # Deliberately use a different N/replicate; these are correctness fixtures,
    # never preflight/main observations or replacement scientific runs.
    task = dict(
        case=case,
        n=32,
        replicate=999,
        phase=phase,
        namespace=[
            2 if phase == "preflight" else 3,
            ["spherical", "correlated", "four_groups"].index(case),
            32,
            999,
        ],
    )
    generated = make_inputs(task)
    expected = independent_inputs(task)
    assert set(generated) == set(expected)
    for key in generated:
        np.testing.assert_array_equal(generated[key], expected[key])


def test_frozen_counts_namespaces_and_budgets():
    assert len(tasks("preflight")) == 6 and len(tasks("main")) == 60
    assert {tuple(t["namespace"]) for t in tasks("preflight")}.isdisjoint(
        tuple(t["namespace"]) for t in tasks("main")
    )
    assert SETTINGS["main_baseline_searches"] == 60 * 4 + 12
    assert SETTINGS["preflight_baseline_searches"] == 6 * 4 + 6
    assert SETTINGS["main_oracle_starts"] == 40 * 4
    assert SETTINGS["total_iteration_cap"] == 10000
    with pytest.raises(ValueError):
        tasks("replacement")


def test_known_law_initialization_is_transformed_and_not_a_selected_partition():
    p = oracle_initial(np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 3.0, 4.0, 5.0]), 1e-4)
    np.testing.assert_allclose(p.weights, 0.25)
    np.testing.assert_allclose(
        p.means[0], (np.array([-3, -3, -1.5, -1.5]) - [1, 2, 3, 4]) / [2, 3, 4, 5]
    )
    assert p.covariances.shape == (4, 4, 4)
    assert np.linalg.eigvalsh(p.covariances).min() >= 1e-4


def test_information_rejection_remains_distinct_from_numerical_failure():
    with pytest.raises(ValueError, match="Too few informative independent units"):
        fit_observed_matrix(np.ones((4, 4)), config=ObservedGMMConfig(("a", "b", "c", "d")))


def test_parameter_roundtrip():
    _, _, initial, _, _ = path_fixture()
    a = {}
    put_parameters(a, "anchor_", initial)
    p = get_parameters(a, "anchor_")
    for name in ("weights", "means", "covariances"):
        np.testing.assert_array_equal(getattr(p, name), getattr(initial, name))


def test_end_to_end_extension_and_oracle_independent_audits(tmp_path):
    from super_deli.benchmarking.observed_recovery import extend_search, oracle_start
    from super_deli.benchmarking.observed_recovery_audit import audit_continuation, audit_oracle
    from super_deli.benchmarking.observed_study import sample, save_fit
    from super_deli.models.observed_gmm import normalization

    x, truth = sample("four_groups", 64, [92004])
    evaluation, _ = sample("four_groups", 32, [92005])
    config = ObservedGMMConfig(tuple(f"trait_{i}" for i in range(4)))
    norm = normalization(x)
    baseline = tmp_path / "baseline"
    save_fit(
        x,
        norm,
        config,
        baseline,
        dict(
            mask="mcar30",
            arm="reference",
            original_n=64,
            task=dict(case="four_groups", key="invented_fixture", n=64),
        ),
        np.arange(64),
        evaluation,
        np.zeros_like(evaluation, dtype=bool),
        truth,
    )
    original = (baseline / "summary.json").read_bytes()
    extended = extend_search(
        baseline, tmp_path / "extended", evaluation, np.zeros_like(evaluation, dtype=bool), truth
    )
    assert audit_continuation(baseline, tmp_path / "extended") == extended["continuation_attempts"]
    oracle = oracle_start(baseline, tmp_path / "oracle")
    assert oracle["excluded_from_selection"] is True
    assert audit_oracle(baseline, tmp_path / "oracle") == 1
    assert (baseline / "summary.json").read_bytes() == original
    # Source/anchor tampering must be caught, rather than silently resumed.
    (baseline / "summary.json").write_bytes(original + b"\n")
    with pytest.raises(AssertionError):
        audit_continuation(baseline, tmp_path / "extended")


def test_component_scope_and_saved_projection_cannot_use_excluded_paths(tmp_path, monkeypatch):
    from super_deli.benchmarking.observed_recovery import extend_search
    from super_deli.io.config import write_json

    x = np.random.default_rng(92201).normal(size=(40, 2))
    x[:10, 0] = np.nan
    config = ObservedGMMConfig(("a", "b"), max_k=2, n_init=1, max_iter=1)
    fitted = fit_observed_matrix(x, config=config)
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    np.savez_compressed(baseline / "fit.npz", **fitted.arrays)
    summary = dict(**fitted.summary, source_indices=fitted.kept_indices.tolist())
    write_json(summary, baseline / "summary.json")
    evaluation = np.zeros((4, 2))
    mask = np.zeros((4, 2), dtype=bool)
    truth = np.zeros(len(x), dtype=int)
    all_paths = extend_search(baseline, tmp_path / "all", evaluation, mask, truth, total_cap=9)
    assert all_paths["continuation_attempts"] == 2

    def no_new_fit(*args, **kwargs):
        raise AssertionError("Projection must reuse saved trajectories without refitting")

    monkeypatch.setattr("super_deli.benchmarking.observed_recovery.fit_start", no_new_fit)
    restricted = extend_search(
        baseline,
        tmp_path / "restricted",
        evaluation,
        mask,
        truth,
        total_cap=9,
        components=(2,),
        saved=tmp_path / "all",
    )
    assert restricted["continuation_attempts"] == 1
    assert restricted["continued_components"] == [2]
    assert restricted["reused_without_refitting"] is True
    assert restricted["candidates"][0]["starts"] == summary["candidates"][0]["starts"]
    assert restricted["status"] == "failed"  # The original unresolved K=1 is preserved.
    with np.load(tmp_path / "restricted/fit.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["k1_s0_final_means"], fitted.arrays["k1_s0_final_means"])
        assert "k1_s0_anchor_means" not in data
    with pytest.raises(ValueError, match="component scope"):
        extend_search(
            baseline,
            tmp_path / "bad",
            evaluation,
            mask,
            truth,
            total_cap=9,
            components=(3,),
            saved=tmp_path / "all",
        )
