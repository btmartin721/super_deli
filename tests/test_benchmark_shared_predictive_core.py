"""Dependency-free paired design, opt-in scope, storage and legacy contracts."""

from dataclasses import replace
from pathlib import Path

import pytest

from super_deli.io.config import read_json
from super_deli.models.grouped_likelihood_config import GroupedLikelihoodConfig
from super_deli.models.predictive_config import PredictiveScreenConfig
from super_deli.models.shared_predictive_config import SharedPredictiveConfig

ROOT = Path(__file__).resolve().parents[1]


def configuration(name="study.json"):
    return SharedPredictiveConfig.from_dict(read_json(ROOT / "examples/shared-predictive" / name))


def test_frozen_design_and_disjoint_pair_seeds():
    c = configuration()
    assert SharedPredictiveConfig.from_dict(c.to_dict()) == c
    assert c.probe() == configuration("preflight.json")
    assert len(c.plan()) == 350 and sum(r["planned_fits"] for r in c.plan()) + 150 == 4350
    assert len(c.probe().plan()) == 14 and sum(r["planned_fits"] for r in c.probe().plan()) == 168
    assert sum(cell.role != "stress" for cell in c.study.cells) * 4 == 40
    assert c.study.model.n_init == 10 and c.study.model.random_seed == 71323
    seeds = [r[k] for r in c.plan() for k in ("data_seed", "test_seed", "split_seed")]
    other = [r[k] for r in c.probe().plan() for k in ("data_seed", "test_seed", "split_seed")]
    assert len(seeds) == len(set(seeds)) == 1050 and set(seeds).isdisjoint(other)
    assert sorted(c.plan(), key=str) == sorted(
        replace(c, study=replace(c.study, cells=tuple(reversed(c.study.cells)))).plan(), key=str
    )


def test_only_covariance_changes_between_families_and_defaults_serialize_unchanged():
    c = configuration()
    full, tied = c.family("full"), c.family("tied")
    a, b = full.to_dict(), tied.to_dict()
    assert b["model"].pop("covariance_type") == "tied" and a == b
    assert "covariance_type" not in a["model"]
    assert full.model.selection_criterion == tied.model.selection_criterion == "aic"
    assert replace(full.model, selection_criterion="bic").selection_criterion == "bic"
    PredictiveScreenConfig().validate_model(tied.model)


@pytest.mark.parametrize(
    "field,value", [("schema_version", True), ("schema_version", 2), ("study", None)]
)
def test_invalid_paired_declarations(field, value):
    raw = configuration().to_dict()
    raw[field] = value
    with pytest.raises((ValueError, TypeError)):
        SharedPredictiveConfig.from_dict(raw)


def test_unsupported_combinations_and_significance_remain_rejected():
    c = configuration()
    with pytest.raises(ValueError):
        c.family("diag")
    with pytest.raises(ValueError):
        SharedPredictiveConfig(c.family("tied"))
    with pytest.raises(ValueError, match="stability"):
        replace(c.family("tied").model, n_bootstrap=1)
    with pytest.raises(ValueError):
        replace(c, study=replace(c.study, screening=PredictiveScreenConfig(n_folds=3)))
    from super_deli.models.predictive_recovery_config import PredictiveRecoveryConfig

    with pytest.raises(ValueError, match="full-family"):
        PredictiveRecoveryConfig(c.family("tied"))
    raw = read_json(ROOT / "examples/grouped-likelihood/smoke.json")["calibration"]
    raw["model"]["covariance_type"] = "tied"
    with pytest.raises(ValueError, match="full-covariance"):
        GroupedLikelihoodConfig.from_dict(raw)


def test_main_requires_matching_measured_preflight(tmp_path):
    from super_deli.benchmarking.shared_predictive_study import run_study

    with pytest.raises(ValueError, match="Measured storage"):
        run_study(configuration(), output_dir=tmp_path / "unlaunched")
    assert not (tmp_path / "unlaunched").exists()


@pytest.mark.parametrize("condition", ["pass", "space", "cap", "failed"])
def test_measured_gate_is_explicit_and_retains_failed_result(tmp_path, monkeypatch, condition):
    import json
    from types import SimpleNamespace

    from super_deli.benchmarking import shared_predictive_cli as gate
    from super_deli.io.config import write_json

    c = configuration()
    if condition == "cap":
        c = replace(c, study=replace(c.study, max_artifact_bytes=128 * 1024**2))
    probe = c.probe()
    source, audit = tmp_path / "probe", tmp_path / "audit"
    source.mkdir()
    audit.mkdir()
    write_json(probe.to_dict(), source / "configuration.json")
    write_json(
        {
            "successful_fits": 167 if condition == "failed" else 168,
            "failed_fits": 0,
            "unattempted_fits": 0,
            "screening_withheld": 0,
        },
        audit / "summary.json",
    )
    (source / "observations.jsonl").write_text(
        "\n".join(json.dumps({**r, "size_bytes": 10000}) for r in probe.plan())
    )
    monkeypatch.setattr(
        gate.shutil,
        "disk_usage",
        lambda p: SimpleNamespace(free=(7 if condition == "space" else 20) * 1024**3),
    )
    output = tmp_path / "gate.json"
    args = dict(probe_study=source, probe_audit=audit, artifact_root=tmp_path, output_file=output)
    if condition == "pass":
        result = gate.storage_preflight(c, **args)
        assert result["status"] == "passed" and result["main_searches_including_replays"] == 4350
    elif condition == "failed":
        with pytest.raises(ValueError, match="audited probe"):
            gate.storage_preflight(c, **args)
        assert not output.exists()
    else:
        with pytest.raises(RuntimeError, match="Storage gate failed"):
            gate.storage_preflight(c, **args)
        assert read_json(output)["status"] == "failed"
