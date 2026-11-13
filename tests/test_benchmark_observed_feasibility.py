"""Independent tree walks, exact operating targets and source isolation."""

import pytest

# ruff: noqa: E402

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from super_deli.benchmarking.observed_feasibility_audit import audit_forest, selection_reference
from super_deli.benchmarking.observed_feasibility_config import ObservedFeasibilityStudy
from super_deli.benchmarking.observed_feasibility_data import measured, seed_registry
from super_deli.benchmarking.observed_feasibility_report import focal_flag
from super_deli.benchmarking.observed_feasibility_run import detector_fit
from super_deli.benchmarking.observed_feasibility_tree import (
    choose_threshold,
    fit_detector,
    native_matrix,
    threshold_counts,
)


def invented():
    rng = np.random.default_rng(761109)
    result = {}
    for i, (g, n) in enumerate((("known", 20), ("near", 10), ("far", 10))):
        x = rng.normal(size=(n, 12))
        x[:, 0] += -2 if i == 0 else 2
        y = np.tile(np.arange(4), 5) if i == 0 else np.full(n, 4)
        result["train_" + g] = dict(morphology=x[:, :4], ecology=x[:, 4:], truth=y)
    return result


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    folder = tmp_path_factory.mktemp("invented_forest")
    data = invented()
    detector_fit(data, folder)
    return folder, data


def test_every_native_stage_loss_and_numeric_tree_roundtrip(fitted):
    folder, data = fitted
    assert audit_forest(folder, data) == 4000
    with np.load(folder / "forest.npz", allow_pickle=False) as z:
        assert len(z["offsets"]) == 101
        assert all(z[k].dtype.kind != "O" for k in z.files)


def test_corruption_is_a_guard_failure(fitted, tmp_path):
    folder, data = fitted
    import shutil

    shutil.copy2(folder / "native-scores.npz", tmp_path / "native-scores.npz")
    with np.load(folder / "forest.npz", allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    a["offsets"][0] = -1
    np.savez_compressed(tmp_path / "forest.npz", **a)
    with pytest.raises(ValueError):
        audit_forest(tmp_path, data)


@pytest.mark.parametrize(
    "x", [np.zeros((3, 11)), np.empty((0, 12)), np.full((2, 12), np.nan), np.full((2, 12), 1e100)]
)
def test_invalid_native_features(x):
    with pytest.raises((ValueError, FloatingPointError)):
        native_matrix(x)


def test_labels_balance_and_feature_whitelist():
    x = np.zeros((20, 12))
    with pytest.raises(ValueError):
        fit_detector(x, np.zeros(20, dtype=int))
    with pytest.raises(ValueError):
        fit_detector(x, np.r_[np.zeros(11, dtype=int), np.ones(9, dtype=int)])
    with pytest.raises(ValueError):
        fit_detector(x, np.r_[np.zeros(10), np.ones(10)])
    d = invented()["train_known"]
    expected = np.concatenate((d["morphology"], d["ecology"]), axis=1)
    d.update(size=np.full(20, 999), latent_morphology=np.full((20, 2), -999), truth=np.arange(20))
    np.testing.assert_array_equal(measured(d), expected)


def test_ties_and_calibration_only_selection():
    counts = threshold_counts(np.array([0.0, 0.5, 0.5, 1.0]))
    assert (
        counts["accepted"][0],
        counts["accepted"][50],
        counts["accepted"][51],
        counts["accepted"][100],
    ) == (4, 3, 1, 1)
    known = threshold_counts(np.full(10, 0.9), np.ones(10, dtype=bool))
    unknown = threshold_counts(np.full(10, 0.1))
    choice = choose_threshold(known, unknown, unknown)
    assert choice == dict(status="selected", index=90, threshold=0.9)
    assert selection_reference(dict(iid=known, near=unknown, far=unknown)) == choice
    # Calibration input is sufficient; no evaluation labels/scores enter the signature.
    import inspect

    assert list(inspect.signature(choose_threshold).parameters) == ["known", "near", "far"]


def test_no_qualifying_point_has_no_fallback():
    known = threshold_counts(np.full(10, 0.5), np.ones(10, dtype=bool))
    unknown = threshold_counts(np.full(10, 0.5))
    assert choose_threshold(known, unknown, unknown) == dict(
        status="available_no_operating_point", index=None, threshold=None
    )
    with pytest.raises(ValueError):
        threshold_counts(np.array([np.nan]))
    with pytest.raises(ValueError):
        threshold_counts(np.array([0.3]), np.array([1]))


def test_test_cohorts_cannot_change_forest_or_training_loss(fitted, tmp_path):
    folder, data = fitted
    new = {k: {n: v.copy() for n, v in d.items()} for k, d in data.items()}
    new["test_iid"] = {k: v.copy() for k, v in data["train_known"].items()}
    new["test_iid"]["morphology"] *= 100
    detector_fit(new, tmp_path)
    with (
        np.load(folder / "forest.npz", allow_pickle=False) as a,
        np.load(tmp_path / "forest.npz", allow_pickle=False) as b,
    ):
        assert set(a.files) == set(b.files)
        for k in a.files:
            np.testing.assert_array_equal(a[k], b[k])


def test_registry_is_materialized_disjoint_and_complete():
    registry = seed_registry(ObservedFeasibilityStudy())
    assert (
        registry["cohorts"] == 902 and registry["seed_paths"] == 4510 and registry["rows"] == 720896
    )
    assert registry["materialized_before_generation"]
    assert all(
        seed[:2] == [20261112, 750] for r in registry["records"] for seed in r["seeds"].values()
    )


def test_focal_stopping_rule_does_not_choose_best_other_cell():
    observations = []
    for i in range(10):
        row = dict(
            arm="residual_0",
            status="available",
            selection=dict(status="selected"),
            joint_targets=dict(diagnostic=i < 5, local=False),
        )
        observations.append(dict(task=dict(case="combined", n=256), assessments=[row]))
    assert focal_flag(observations)["flag"]
    observations[-1]["task"]["case"] = "linear_standard"
    assert not focal_flag(observations)["flag"]


@pytest.mark.parametrize("case", ["linear_weak", "linear_standard", "combined", "confounded_size"])
def test_independent_raw_laws_for_all_invented_cases(case, tmp_path):
    from super_deli.benchmarking.abstention_config import CASES
    from super_deli.benchmarking.observed_feasibility_audit import raw_roles
    from super_deli.benchmarking.observed_feasibility_config import roles
    from super_deli.benchmarking.observed_feasibility_data import generate_role, role_ids
    from super_deli.io.config import write_json

    task = dict(
        key="invented_" + case,
        case=case,
        n=32,
        replicate=0,
        fixture=True,
        seed_path=[20261112, 750, 0, CASES.index(case), 32, 0],
    )
    ids = {}
    for role in roles(task):
        name = role["name"]
        np.savez_compressed(tmp_path / (name + ".npz"), **generate_role(task, name))
        ids[name] = role_ids(task, name)
    write_json(ids, tmp_path / "identities.json")
    actual, seen = raw_roles(tmp_path, task)
    assert len(actual) == (11 if case == "confounded_size" else 10)
    assert len(seen) == sum(len(v) for v in ids.values())


@pytest.mark.parametrize(
    "fault", ["none", "head", "duplicate", "missing", "incomplete", "top_head"]
)
def test_source_completion_requires_all_six_exact_head_jobs(fault):
    from super_deli.benchmarking.observed_feasibility_source import SOURCE_HEAD, check_completion

    complete = dict(completed=True, head_sha=SOURCE_HEAD, ci_jobs=6)
    ci = dict(
        head_sha=SOURCE_HEAD,
        status="completed",
        conclusion="success",
        jobs=[
            dict(
                name=f"{kind} ({v})", head_sha=SOURCE_HEAD, status="completed", conclusion="success"
            )
            for kind in ("checks", "genomics", "neural")
            for v in ("3.11", "3.12")
        ],
    )
    if fault == "head":
        ci["jobs"][0]["head_sha"] = "0" * 40
    if fault == "duplicate":
        ci["jobs"][0]["name"] = ci["jobs"][1]["name"]
    if fault == "missing":
        ci["jobs"].pop()
    if fault == "incomplete":
        ci["jobs"][0]["status"] = "in_progress"
    if fault == "top_head":
        ci["head_sha"] = "0" * 40
    if fault == "none":
        check_completion(complete, ci)
    else:
        with pytest.raises(ValueError):
            check_completion(complete, ci)


@pytest.mark.parametrize("stages", [99, 100])
def test_native_stage_nonfinite_or_incomplete_is_rejected(stages):
    from super_deli.benchmarking.observed_feasibility_tree import native_stages

    class InvalidNative:
        def staged_decision_function(self, x):
            for _ in range(stages):
                yield np.full(len(x), np.nan if stages == 100 else 0.0)

        def predict_proba(self, x):
            return np.full((len(x), 2), 0.5)

    with pytest.raises(ValueError, match="complete native tree stages"):
        native_stages(InvalidNative(), np.zeros((3, 12)))
