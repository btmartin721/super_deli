"""Grouped orchestration schemas never broaden native inferential capabilities."""

from dataclasses import replace

import pytest

from super_deli.benchmarking.grouped_analysis_study import SETTINGS, model_arms, tasks
from super_deli.models.grouped_analysis_config import GroupedAnalysisArm, GroupedAnalysisConfig
from super_deli.models.predictive_bootstrap_config import PredictiveBootstrapConfig
from super_deli.models.predictive_config import PredictiveScreenConfig
from super_deli.models.predictive_fold_size import PredictiveFoldSizePolicy
from super_deli.models.tabular_analysis_config import TabularAnalysisConfig


def test_roundtrip_and_schema_isolation():
    c = model_arms(2)
    assert GroupedAnalysisConfig.from_dict(c.to_dict()) == c
    with pytest.raises(ValueError):
        TabularAnalysisConfig.from_dict(c.to_dict())
    assert len(c.arms) == 6 and len({a.signature() for a in c.arms}) == 1


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"schema_version": True, "arms": []},
        {"schema_version": 2, "arms": []},
        {"schema_version": 1, "arms": []},
        {"schema_version": 1, "arms": [], "other": 1},
        {"schema_version": 1, "arms": "bad"},
    ],
)
def test_invalid_top_level(value):
    with pytest.raises(ValueError):
        GroupedAnalysisConfig.from_dict(value)


@pytest.mark.parametrize("name", ["", "../escape", "Bad", "a/b", "a b", None, "a" * 65])
def test_safe_names(name):
    with pytest.raises(ValueError):
        replace(model_arms().arms[0], name=name)


@pytest.mark.parametrize(
    "change",
    [
        {"assignment_threshold": 0.5},
        {"n_threads": 2},
        {"missing_views": "complete_cases"},
        {"target_level": "specimen"},
    ],
)
def test_disallowed_point_policies(change):
    with pytest.raises(ValueError):
        a = model_arms().arms[0]
        replace(a, model=replace(a.model, **change))


@pytest.mark.parametrize(
    "change",
    [
        {"pca_fit": "source_rows"},
        {"aggregation": "none"},
        {"feature_names": ()},
        {"scaling": "center"},
        {"missing_policy": "mean"},
    ],
)
def test_disallowed_view_policies(change):
    with pytest.raises(ValueError):
        a = model_arms().arms[0]
        v = replace(a.model.views[0], **change)
        replace(a, model=replace(a.model, views=(v, a.model.views[1])))


@pytest.mark.parametrize(
    "field,value", [("n_pcs", 3), ("rank_tol", 1e-9), ("target_level", "population")]
)
def test_common_preprocessing(field, value):
    c = model_arms()
    a = replace(c.arms[0], model=replace(c.arms[0].model, **{field: value}))
    with pytest.raises(ValueError):
        replace(c, arms=(a, c.arms[1]))


def test_stability_and_scaled_policy_boundaries():
    c = model_arms()
    bootstrap = PredictiveBootstrapConfig(2, 71)
    for i in (0, 1, 2, 3, 4):
        with pytest.raises(ValueError):
            replace(c.arms[i], bootstrap=bootstrap)
    with pytest.raises(ValueError):
        replace(c.arms[3], model=replace(c.arms[3].model, n_bootstrap=2))
    assert replace(c.arms[2], model=replace(c.arms[2].model, n_bootstrap=2))
    assert replace(c.arms[5], bootstrap=bootstrap, fold_size_policy=PredictiveFoldSizePolicy())
    for a in model_arms(target_level="population").arms:
        if a.kind == "predictive":
            with pytest.raises(ValueError):
                replace(a, bootstrap=bootstrap)
    with pytest.raises(ValueError):
        replace(c.arms[5], screen=PredictiveScreenConfig(n_folds=4), bootstrap=bootstrap)
    with pytest.raises(ValueError):
        replace(c.arms[4], fold_size_policy=PredictiveFoldSizePolicy())


def test_invalid_kind_screen_and_duplicate_names():
    c = model_arms()
    with pytest.raises(ValueError):
        replace(c.arms[0], kind="other")
    with pytest.raises(ValueError):
        replace(c.arms[0], screen=PredictiveScreenConfig())
    with pytest.raises(ValueError):
        replace(c.arms[5], screen=None)
    with pytest.raises(ValueError):
        replace(c, arms=(c.arms[0], c.arms[0]))
    with pytest.raises(ValueError):
        replace(c, arms=tuple(replace(c.arms[0], name=f"a{i}") for i in range(17)))
    with pytest.raises(ValueError):
        GroupedAnalysisArm.from_dict({"name": "a", "kind": "ordinary", "model": {}, "extra": 1})


def test_frozen_search_counts_and_disjoint_streams():
    main, preflight = tasks("main"), tasks("preflight")

    def count(rows):
        return sum(16 + 6 * t["draws"] + 16 * int(t["replay"]) for t in rows)

    assert len(main) == 36 and count(main) == 2568
    assert len(preflight) == 2 and count(preflight) == 364
    assert {t["data_seed"] for t in main}.isdisjoint(t["data_seed"] for t in preflight)
    assert SETTINGS["artifact_cap_bytes"] == 2 * 1024**3
    with pytest.raises(ValueError):
        tasks("other")


def test_guarded_validation_runner(monkeypatch):
    import runpy
    from pathlib import Path

    monkeypatch.setattr("sys.argv", ["test", "--invalid"])
    result = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples/grouped-analysis/validate.py"),
        run_name="import_only",
    )
    assert callable(result["main"])


@pytest.mark.parametrize("cap", [True, 0, "bad"])
def test_invalid_budget_override(cap, tmp_path):
    from super_deli.benchmarking.grouped_analysis_study import storage_gate

    with pytest.raises(ValueError, match="positive integer"):
        storage_gate(
            tmp_path,
            tmp_path / "audit.json",
            tmp_path,
            tmp_path / "gate.json",
            artifact_cap_bytes=cap,
        )
