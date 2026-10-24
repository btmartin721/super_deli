"""Dependency-free diagnostic gates, fixed budgets and storage boundaries."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from super_deli.benchmarking.predictive_recovery_gates import decompose_gates
from super_deli.io.config import read_json, write_json
from super_deli.models.predictive_recovery_config import PredictiveRecoveryConfig

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return PredictiveRecoveryConfig.from_dict(
        read_json(ROOT / "examples/predictive-recovery/study.json")
    )


def c(k, aic=100, status="eligible"):
    return {"k": k, "aic": aic, "status": status}


def folds():
    return [
        {
            "status": "complete",
            "validation_indices": indices,
            "candidates": [c(1), c(2, 90), c(4, 80)],
            "log_densities": {
                "1": [0.0] * len(indices),
                "2": [-1.0] * len(indices),
                "4": [2.0] * len(indices),
            },
        }
        for indices in ([0], [1, 2, 3])
    ]


def test_separate_eligibility_and_prediction_gate_removals():
    fs = folds()
    fs[0]["candidates"][2]["status"] = "small_component"
    r = decompose_gates([c(1), c(2, 90), c(4, 80)], fs, 4)
    assert (r["ordinary_k"], r["eligibility_only_k"], r["gain_only_k"], r["actual_k"]) == (
        4,
        2,
        4,
        1,
    )
    assert r["candidates"][2]["pooled_gain_all_fitted_folds"] == 2
    assert "never production" in r["role"]


def test_full_eligibility_is_retained_in_every_counterfactual():
    r = decompose_gates([c(1), c(2, 90), c(4, 80, "small_component")], folds(), 4)
    assert r["gain_only_k"] == r["actual_k"] == 1
    assert r["eligibility_only_k"] == 2


def test_gain_pooling_weights_localities_and_preserves_strict_zero():
    fs = folds()
    fs[0]["log_densities"]["2"] = [3.0]
    fs[1]["log_densities"]["2"] = [-1.0] * 3
    r = decompose_gates([c(1), c(2, 90)], fs, 4)
    assert r["actual_k"] == 1 and r["candidates"][1]["pooled_gain_all_fitted_folds"] == 0
    fs[0]["log_densities"]["2"] = [3.0000001]
    assert decompose_gates([c(1), c(2, 90)], fs, 4)["actual_k"] == 2


@pytest.mark.parametrize("issue", ["failed", "nonfinite", "missing", "unresolved"])
def test_numerical_failure_is_not_a_one_component_diagnostic(issue):
    fs = folds()
    if issue == "failed":
        fs[0]["status"] = "failed"
    elif issue == "nonfinite":
        fs[0]["log_densities"]["4"] = [float("nan")]
    elif issue == "missing":
        del fs[0]["log_densities"]["4"]
    else:
        fs[0]["candidates"][2]["status"] = "nonconverged"
    result = decompose_gates([c(1), c(2, 90), c(4, 80)], fs, 4)
    assert result["status"] == "withheld" and result["actual_k"] is None


def test_partition_overlap_rejected():
    fs = folds()
    fs[1]["validation_indices"] = [0, 2, 3]
    with pytest.raises(ValueError):
        decompose_gates([c(1), c(2, 90), c(4, 80)], fs, 4)


def test_frozen_design_budget_and_seed_independence():
    c = configuration()
    plan = c.plan()
    probe = c.probe()
    assert PredictiveRecoveryConfig.from_dict(c.to_dict()) == c
    assert len(plan) == 100 and sum(r["planned_searches"] for r in plan) + 150 == 3350
    assert sum(r["planned_oracles"] for r in plan) == 450
    assert sum(r["planned_searches"] for r in probe.plan()) == 128
    assert sum(r["planned_oracles"] for r in probe.plan()) == 18
    assert probe == PredictiveRecoveryConfig.from_dict(
        read_json(ROOT / "examples/predictive-recovery/preflight.json")
    )
    seeds = [s for r in plan for s in [r["data_seed"], r["test_seed"], *r["partition_seeds"]]]
    assert len(seeds) == len(set(seeds)) == 700
    assert set(seeds).isdisjoint(
        s for r in probe.plan() for s in [r["data_seed"], r["test_seed"], *r["partition_seeds"]]
    )
    reordered = replace(c, study=replace(c.study, cells=tuple(reversed(c.study.cells))))
    assert sorted(plan, key=str) == sorted(reordered.plan(), key=str)
    assert all("planned_fits" not in r for r in plan)


@pytest.mark.parametrize(
    "update",
    [
        {"partitions": True},
        {"partitions": 1},
        {"partitions": 6},
        {"extended_starts": 10},
        {"extended_starts": 101},
        {"schema_version": True},
    ],
)
def test_bad_design_rejected(update):
    with pytest.raises(ValueError):
        replace(configuration(), **update)


def test_full_study_requires_measured_gate_before_creating_output(tmp_path):
    from super_deli.benchmarking.predictive_recovery_study import run_study

    p = tmp_path / "not-created"
    with pytest.raises(ValueError, match="preflight"):
        run_study(configuration(), output_dir=p)
    assert not p.exists()


@pytest.mark.parametrize("problem", ["none", "free", "cap", "incomplete"])
def test_measured_storage_gate(tmp_path, monkeypatch, problem):
    import super_deli.benchmarking.predictive_recovery_preflight as preflight

    c = configuration()
    probe = tmp_path / "probe"
    audit = tmp_path / "audit"
    probe.mkdir()
    audit.mkdir()
    write_json(c.probe().to_dict(), probe / "configuration.json")
    write_json(
        {
            "successful_searches": 128,
            "successful_oracles": 18 if problem != "incomplete" else 17,
            "withheld_parts": 0,
        },
        audit / "summary.json",
    )
    (probe / "observations.jsonl").write_text(
        "".join(
            __import__("json").dumps({"cell": x.name, "size_bytes": 1024}) + "\n"
            for x in c.study.cells
        )
    )
    monkeypatch.setattr(
        preflight,
        "retained_bytes",
        lambda root: c.study.max_artifact_bytes if problem == "cap" else 0,
    )
    monkeypatch.setattr(
        preflight.shutil,
        "disk_usage",
        lambda root: SimpleNamespace(free=0 if problem == "free" else 100 * 1024**3),
    )
    if problem == "none":
        r = preflight.storage_preflight(
            c,
            probe_study=probe,
            probe_audit=audit,
            artifact_root=tmp_path,
            output_file=tmp_path / "gate.json",
        )
        assert (
            r["status"] == "passed"
            and r["main_searches_including_replays"] == 3350
            and r["main_single_k_fits"] == 450
        )
    else:
        with pytest.raises((ValueError, RuntimeError)):
            preflight.storage_preflight(
                c,
                probe_study=probe,
                probe_audit=audit,
                artifact_root=tmp_path,
                output_file=tmp_path / "gate.json",
            )
