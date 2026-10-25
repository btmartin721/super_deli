"""Exact exposure arithmetic, configuration boundaries and unchanged defaults."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from super_deli.models.integrative_config import IntegrativeConfig
from super_deli.models.predictive_bootstrap_study import PredictiveBootstrapStudy
from super_deli.models.predictive_config import PredictiveScreenConfig
from super_deli.models.predictive_fold_size import PredictiveFoldMixture, PredictiveFoldSizePolicy
from super_deli.models.scaled_fold_study import ScaledFoldStudy

ROOT = Path(__file__).resolve().parents[1]


def config(name="study"):
    return ScaledFoldStudy.from_dict(
        json.loads((ROOT / f"examples/scaled-fold-stability/{name}.json").read_text())
    )


@pytest.mark.parametrize(
    "u,t,expected",
    [(64, 51, 4), (64, 52, 5), (256, 204, 4), (256, 205, 5), (40, 32, 4), (40, 33, 5), (100, 1, 2)],
)
def test_exact_thresholds(u, t, expected):
    evidence = PredictiveFoldSizePolicy().evidence(5, range(u), range(t))
    assert evidence["training_minimum"] == expected
    assert evidence["full_distinct"] == u and evidence["training_distinct"] == t
    assert evidence["full_minimum"] == 5


def test_copies_do_not_inflate_exposure_and_values_are_not_inputs():
    p = PredictiveFoldSizePolicy()
    a = p.evidence(5, list(range(40)) * 3, list(range(32)) * 4)
    b = p.evidence(5, range(40), range(32))
    assert a == b and a["training_minimum"] == 4


@pytest.mark.parametrize(
    "minimum,full,train",
    [
        (True, [0, 1], [0]),
        (1, [0, 1], [0]),
        (5, [], []),
        (5, [0, 1], []),
        (5, [0, 1], [0, 1]),
        (5, [0, 1], [2]),
    ],
)
def test_invalid_exposure_is_not_repaired(minimum, full, train):
    with pytest.raises(ValueError):
        PredictiveFoldSizePolicy().evidence(minimum, full, train)


@pytest.mark.parametrize(
    "value",
    [
        {},
        dict(schema_version=1),
        dict(schema_version=True, policy="proportional_distinct"),
        dict(schema_version=2, policy="proportional_distinct"),
        dict(schema_version=1, policy="fixed"),
        dict(schema_version=1, policy="proportional_distinct", minimum=4),
    ],
)
def test_policy_schema_strict(value):
    with pytest.raises(ValueError):
        PredictiveFoldSizePolicy.from_dict(value)


def test_budgets_and_fresh_probe_namespace():
    c = config()
    p = c.probe()
    assert sum(r["planned_fits"] for r in c.plan()) == 48000
    assert sum(r["planned_fits"] for r in p.plan()) == 9600
    assert c == ScaledFoldStudy.from_dict(c.to_dict())
    main = {r[k] for r in c.plan() for k in ("data_seed", "split_seed", "bootstrap_seed")}
    probe = {r[k] for r in p.plan() for k in ("data_seed", "split_seed", "bootstrap_seed")}
    assert main.isdisjoint(probe)
    assert c.max_artifact_bytes == 6 * 1024**3


def test_legacy_schema_and_full_model_guard_unchanged():
    old = json.loads((ROOT / "examples/predictive-stability/study.json").read_text())
    c = PredictiveBootstrapStudy.from_dict(old)
    assert "fold_size_policy" not in c.to_dict()
    with pytest.raises(ValueError):
        PredictiveBootstrapStudy.from_dict(config().to_dict())
    with pytest.raises(ValueError, match="min_cluster_size"):
        replace(c.model, min_cluster_size=4)
    policy = PredictiveFoldSizePolicy()
    policy.validate_model(c.model, c.screening)
    effective = PredictiveFoldMixture.from_model(c.model, 4)
    assert effective.min_cluster_size == 4 and c.model.min_cluster_size == 5
    assert "fold_size_policy" not in c.model.to_dict()
    with pytest.raises(ValueError):
        IntegrativeConfig.from_dict(dict(c.model.to_dict(), fold_size_policy=policy.to_dict()))


@pytest.mark.parametrize(
    "field,value", [("covariance_type", "full"), ("target_level", "population")]
)
def test_unsupported_families_or_targets(field, value):
    c = config()
    model = replace(c.model, **{field: value})
    with pytest.raises(ValueError):
        c.fold_size_policy.validate_model(model, c.screening)


def test_unsupported_folds_and_effective_minima():
    c = config()
    with pytest.raises(ValueError):
        c.fold_size_policy.validate_model(c.model, PredictiveScreenConfig(n_folds=4))
    for m in (1, True, 6):
        with pytest.raises(ValueError):
            PredictiveFoldMixture.from_model(c.model, m)


@pytest.mark.parametrize("budget,passed", [(6 * 1024**3, True), (1024**2, False)])
def test_measured_probe_gate_requires_both_audits_and_retains_failure(
    tmp_path, monkeypatch, budget, passed
):
    from super_deli.benchmarking import scaled_fold_preflight as module
    from super_deli.io.config import read_json, write_json

    c = replace(config(), max_artifact_bytes=budget)
    probe = tmp_path / "probe"
    probe.mkdir()
    (tmp_path / "audit").mkdir()
    write_json(c.probe().to_dict(), probe / "configuration.json")
    rows = []
    for i, cell in enumerate(c.cells):
        write_json({}, probe / f"manifest_{i}.json")
        rows.append(dict(cell=cell.name, size_bytes=10000, manifest=f"manifest_{i}.json"))
    (probe / "observations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_json(
        dict(status="passed", audited_searches=9600, unattempted_fits=0, failed_draws=0),
        tmp_path / "audit/summary.json",
    )
    write_json(dict(status="verified", audited_searches=9600), tmp_path / "check.json")
    monkeypatch.setattr(
        module.shutil, "disk_usage", lambda p: type("Usage", (), dict(free=100 * 1024**3))()
    )
    target = tmp_path / "gate.json"
    kwargs = dict(
        probe_study=probe,
        probe_audit=tmp_path / "audit",
        probe_check=tmp_path / "check.json",
        artifact_root=tmp_path,
        output_file=target,
    )
    if passed:
        result = module.storage_preflight(c, **kwargs)
        assert result["status"] == "passed" and result["main_searches_including_replays"] == 48180
        assert result["replay_archive_allowance"] > 0
    else:
        with pytest.raises(RuntimeError, match="Storage gate failed"):
            module.storage_preflight(c, **kwargs)
        assert read_json(target)["status"] == "failed"
