"""Independent dense-likelihood, exclusion and parent-level uncertainty tests."""

# ruff: noqa: E402
import copy

import pytest

np = pytest.importorskip("numpy")
from scipy.stats import binomtest, multivariate_normal

from super_deli.benchmarking.known_reference_config import KnownReferenceStudy, roles, tasks
from super_deli.benchmarking.known_reference_data import (
    fold_data,
    generate,
    matrix,
    seed_registry,
    training_data,
)
from super_deli.benchmarking.known_reference_report import paired, parent_outcome
from super_deli.benchmarking.local_reference_fixture import matrices
from super_deli.models.matched_preprocessing import prepare_views
from super_deli.models.reference_abstention import decide, distances, fit_support, transform
from super_deli.models.reference_density import (
    calibrate_density_support,
    density_scores,
    fit_density_support,
)


def fitted():
    x, y, xc, yc, _ = matrices()
    x = x.copy()
    x[y == 2, :4] *= 4
    _, prep = prepare_views(x, 4)
    return x, y, xc, yc, prep, fit_density_support(x, y, prep, 4)


def test_class_specific_quadratics_gaussian_likelihood_and_fixed_rank():
    x, y, xc, yc, prep, a = fitted()
    score = density_scores(xc, a)
    z = transform(xc, a)
    for c in range(4):
        delta = z - a["centers"][c]
        cov = a["covariances"][c]
        direct = np.einsum("ni,ij,nj->n", delta, np.linalg.inv(cov), delta)
        np.testing.assert_allclose(score[:, c], direct, rtol=1e-11, atol=1e-11)
        nll = -multivariate_normal.logpdf(z, mean=a["centers"][c], cov=cov)
        np.testing.assert_allclose(
            nll, 0.5 * (direct + a["log_determinants"][c] + z.shape[1] * np.log(2 * np.pi))
        )
    c = calibrate_density_support(a, xc, yc)
    np.testing.assert_array_equal(c["calibration_ranks"], [20] * 4)
    np.testing.assert_allclose(c["thresholds"], [max(score[yc == k, k]) for k in range(4)])
    pooled = fit_support(x, y, prep, 4)
    assert not np.allclose(distances(xc, pooled), score)


def test_shared_class_covariances_reduce_to_pooled_scores():
    rng = np.random.default_rng(14)
    base = rng.normal(size=(16, 12))
    x = np.vstack([base + c for c in range(3)])
    y = np.repeat(np.arange(3), 16)
    _, prep = prepare_views(x, 4)
    a = fit_density_support(x, y, prep, 3)
    b = fit_support(x, y, prep, 3)
    np.testing.assert_allclose(density_scores(x, a), distances(x, b), rtol=1e-12, atol=1e-12)


def test_density_support_is_frozen_queries_do_not_refit():
    _, _, xc, yc, _, a = fitted()
    before = {k: v.copy() for k, v in a.items()}
    calibrate_density_support(a, xc + 50, yc)
    for k in a:
        np.testing.assert_array_equal(a[k], before[k])
    expected = density_scores(xc, a)
    np.testing.assert_array_equal(density_scores(xc[::-1], a)[::-1], expected)
    np.testing.assert_allclose(density_scores(xc[:1], a), expected[:1], rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="memory"):
        density_scores(xc, a, max_working_bytes=128)
    with pytest.raises(ValueError, match="integer"):
        density_scores(xc, a, max_working_bytes=True)
    with pytest.raises(ValueError):
        density_scores(np.full_like(xc, np.nan), a)


def test_constant_class_and_insufficient_calibration_stay_unavailable():
    x, y, xc, yc, prep, a = fitted()
    x[y == 0] = x[y == 0][0]
    with pytest.raises(ValueError, match="within-class"):
        fit_density_support(x, y, prep, 4)
    with pytest.raises(ValueError, match="twenty"):
        calibrate_density_support(a, xc[:-1], yc[:-1])
    with pytest.raises(ValueError):
        fit_density_support(x, y, prep, True)
    p = np.array([[0.8, 0.2]])
    s = np.array([[0.0, 0.0]])
    assert not decide(p, s, np.zeros(2), "distance")["accepted"][0]


def test_known_only_fit_ignores_excluded_rows_and_evaluation_labels():
    task = tasks(KnownReferenceStudy(fixture=True), "main")[0]
    parent = {k: generate(task, k) for k, _ in roles(task)}
    for h in range(4):
        first = fold_data(parent, h)
        other = copy.deepcopy(parent)
        for role in ("train", "calibration"):
            selected = other[role]["truth"] == h
            other[role]["morphology"][selected] = 1e100
            other[role]["ecology"][selected] = -1e100
        for role in ("iid", "near", "far", "overlap_pool"):
            other[role]["truth"][:] = 4
        # Fitting arrays are selected independently of all evaluation cohorts.
        known = [c for c in range(4) if c != h]
        for role in ("train", "calibration"):
            idx = np.flatnonzero(other[role]["truth"] != h)
            np.testing.assert_array_equal(
                matrix(first["training" if role == "train" else role]), matrix(other[role])[idx]
            )
            assert all(int(v) in known for v in other[role]["truth"][idx])
        _, prep = prepare_views(matrix(first["training"]), 4)
        a = fit_density_support(matrix(first["training"]), first["training"]["truth"], prep, 3)
        second = training_data(other, h)
        changed_x = matrix(second["train"])
        _, changed_prep = prepare_views(changed_x, 4)
        b = fit_density_support(changed_x, second["train"]["truth"], changed_prep, 3)
        first_cal = calibrate_density_support(
            a, matrix(first["calibration"]), first["calibration"]["truth"]
        )
        second_cal = calibrate_density_support(
            b, matrix(second["calibration"]), second["calibration"]["truth"]
        )
        for key in first_cal:
            np.testing.assert_array_equal(first_cal[key], second_cal[key])
        assert len(first["training"]["truth"]) == 24
        assert len(first["calibration"]["truth"]) == 60
        assert len(first["cohorts"]["heldout"]["x"]) == 16


def test_parent_rules_bound_incomplete_folds_without_counting_them_independent():
    def rows(values):
        return [
            dict(
                arm="logistic",
                heldout=h,
                status="available" if v is not None else "failed",
                policy_status={"density": "available"},
                joint_targets={"density": v},
            )
            for h, v in enumerate(values)
        ]

    assert parent_outcome(rows([True, True, True, None]), "logistic", "density") is True
    assert parent_outcome(rows([False, False, None, None]), "logistic", "density") is False
    assert parent_outcome(rows([True, True, None, False]), "logistic", "density") is None
    x = paired([True, False, None], [False, True, False])
    assert (
        x["requested"] == 3 and x["unavailable"] == 1 and x["new_only"] == x["baseline_only"] == 1
    )
    assert x["descriptive_95_interval"][0] <= 0 <= x["descriptive_95_interval"][1]
    # The primary interval utility is independently matched to exact inversion.
    from super_deli.benchmarking.local_reference_report import rate

    r = rate([True] * 7 + [False] * 3)
    ci = binomtest(7, 10).proportion_ci(method="exact")
    np.testing.assert_allclose(
        r["descriptive_95_interval"], [ci.low, ci.high], rtol=1e-10, atol=1e-10
    )


def test_registry_materialized_before_generation_and_disjoint_namespaces():
    r = seed_registry(KnownReferenceStudy())
    assert (r["cohorts"], r["rows"], r["seed_paths"]) == (550, 523776, 2750)
    assert r["materialized_before_generation"]
    assert all(v[:2] == [20261112, 760] for row in r["records"] for v in row["seeds"].values())


def test_no_accepted_known_specimens_preserves_undefined_accuracy():
    from super_deli.benchmarking.abstention_metrics import metrics

    p = np.asarray([[0.9, 0.1], [0.2, 0.8]])
    result = decide(p, np.full((2, 2), 10.0), np.zeros(2), "distance")
    m = metrics(np.asarray([0, 1]), result, known=True)
    assert m["accepted"] == 0 and m["accepted_accuracy"] is None
    assert m["raw_accuracy"] == 1 and m["correct_accepted_fraction"] == 0


def test_class_specific_scores_also_differ_from_fifth_neighbors():
    from super_deli.models.local_reference import fit_local_support, local_distances

    x, y, xc, _, prep, a = fitted()
    pooled = fit_support(x, y, prep, 4)
    local = fit_local_support(x, y, pooled)
    assert not np.allclose(density_scores(xc, a), local_distances(xc, pooled, local))


def test_unavailable_fits_keep_attempted_counts_and_withhold_inference(tmp_path, monkeypatch):
    import super_deli.benchmarking.known_reference_run as native
    from super_deli.benchmarking.known_reference_audit import audit_fold
    from super_deli.benchmarking.known_reference_run import run_fold

    task = tasks(KnownReferenceStudy(fixture=True), "main")[0]
    parent = {k: generate(task, k) for k, _ in roles(task)}

    def fail(arm, *args):
        args[-1].mkdir()
        return dict(
            arm=arm,
            status="failed",
            attempted=1,
            neural_fits=int(arm != "logistic"),
            failed_points=1,
            numerical_guard_failed=False,
            error="Invented unavailable fit",
            completed_optimizer_updates=0,
            seconds=0.0,
        )

    monkeypatch.setattr(native, "supervised_arm", fail)
    folder = tmp_path / "fold"
    rows = run_fold(parent, 0, folder, "invented_unavailable", tmp_path / "STOP.json")
    assert sum(r["attempted"] for r in rows) == 2
    assert all(all(v is None for v in r["joint_targets"].values()) for r in rows)
    audit = audit_fold(parent, 0, folder)
    assert audit["neural_fits"] == 1 and audit["verified_updates"] == 0
    assert audit["policy_cohort_evaluations"] == 0
