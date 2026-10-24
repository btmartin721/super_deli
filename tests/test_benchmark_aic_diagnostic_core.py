"""Dependency-free frozen design and launch safety contracts."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from super_deli.io.config import read_json, write_json
from super_deli.models.aic_diagnostic_config import ARMS, AICDiagnosticConfig

ROOT = Path(__file__).resolve().parents[1]


def configuration(name="study.json"):
    return AICDiagnosticConfig.from_dict(
        read_json(ROOT / "examples/aic-gaussian-diagnostics" / name)
    )


def test_approved_budget_and_legacy_model_are_unchanged():
    c = configuration()
    assert AICDiagnosticConfig.from_dict(json.loads(json.dumps(c.to_dict()))) == c
    assert len(c.plan()) == 600
    assert sum(r["planned_fits"] for r in c.plan()) + 50 == 1250
    assert c.n_test == 2048 and c.model.n_init == 10 and c.long_n_init == 100
    assert c.model.random_seed == 71323 and c.model.min_cluster_size == 5
    assert c.model.n_bootstrap == 0 and c.model.covariance_type == "full"
    assert c.model.views[0].pca_fit == "target_means"
    assert c.max_artifact_bytes == 1024**3 and c.min_free_bytes == 8 * 1024**3
    assert replace(c.arm("starts_100"), n_init=10) == c.model
    assert c.arm("baseline") == c.arm("homogeneous_mean_noise") == c.model


def test_seed_identity_is_order_invariant_and_disjoint():
    c = configuration()
    seeds = [r[k] for r in c.plan() for k in ("data_seed", "test_seed")]
    assert len(seeds) == len(set(seeds)) == 1200
    assert sorted(c.plan(), key=str) == sorted(
        replace(c, cases=tuple(reversed(c.cases)), sizes=(256, 64)).plan(), key=str
    )
    assert set(seeds).isdisjoint(
        r[k] for r in configuration("preflight.json").plan() for k in ("data_seed", "test_seed")
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"n_datasets": True},
        {"n_scores": 0},
        {"n_test": 19},
        {"n_test": 21},
        {"long_n_init": 9},
        {"sizes": (64, 64)},
        {"sizes": (True,)},
        {"cases": ("unknown",)},
        {"cases": ("independent_gaussian",) * 2},
        {"random_seed": -1},
        {"random_seed": True},
        {"schema_version": True},
        {"max_artifact_bytes": 1},
        {"min_free_bytes": False},
    ],
)
def test_invalid_design_is_rejected(updates):
    with pytest.raises(ValueError):
        replace(configuration(), **updates)


def test_unknown_keys_model_variants_and_unknown_arms_rejected():
    c = configuration()
    data = c.to_dict()
    data["unknown"] = 1
    with pytest.raises(ValueError):
        AICDiagnosticConfig.from_dict(data)
    for update in ({"n_bootstrap": 1}, {"n_pcs": 2}, {"covariance_type": "tied"}):
        with pytest.raises(ValueError):
            replace(c, model=replace(c.model, **update))
    with pytest.raises(ValueError):
        c.arm("pick_best")


def test_full_launch_requires_matching_measured_preflight(tmp_path):
    from super_deli.benchmarking.aic_diagnostic_study import run_study

    with pytest.raises(ValueError, match="Measured storage"):
        run_study(configuration(), output_dir=tmp_path / "study")
    path = tmp_path / "preflight.json"
    write_json({"status": "failed", "configuration_sha256": "wrong"}, path)
    with pytest.raises(ValueError, match="differs"):
        run_study(configuration(), output_dir=tmp_path / "study", preflight=path)
    assert not (tmp_path / "study").exists()


def test_probe_budget_and_full_settings_are_preserved():
    from super_deli.benchmarking.aic_diagnostic_preflight import probe_config

    c = configuration()
    probe = probe_config(c)
    assert probe == configuration("preflight.json")
    assert sum(r["planned_fits"] for r in probe.plan()) + 1 == 25
    assert probe.model == c.model and probe.n_test == c.n_test


def test_storage_gate_rejects_unverified_probe(tmp_path):
    from super_deli.benchmarking.aic_diagnostic_preflight import storage_preflight

    study, audit, replay = (tmp_path / p for p in ("probe", "audit", "replay"))
    for p in (study, audit, replay):
        p.mkdir()
    write_json(configuration("preflight.json").to_dict(), study / "configuration.json")
    write_json({"point_fits": 24, "successful_fits": 23}, audit / "summary.json")
    with pytest.raises(ValueError, match="fully successful"):
        storage_preflight(
            configuration(),
            probe_study=study,
            probe_audit=audit,
            probe_replay=replay,
            artifact_root=tmp_path,
            output_file=tmp_path / "gate.json",
        )
    assert not (tmp_path / "gate.json").exists()


def test_arm_names_are_frozen():
    assert ARMS == ("baseline", "homogeneous_mean_noise", "starts_100")


@pytest.mark.parametrize("condition", ["pass", "space", "cap", "replay"])
def test_measured_gate_and_failure_artifact(tmp_path, monkeypatch, condition):
    from types import SimpleNamespace

    from super_deli.benchmarking import aic_diagnostic_preflight as gate

    c = configuration()
    if condition == "cap":
        c = replace(c, max_artifact_bytes=64 * 1024**2)
    probe = gate.probe_config(c)
    study, audit, replay = (tmp_path / p for p in ("probe", "audit", "replay"))
    for p in (study, audit, replay):
        p.mkdir()
    write_json(probe.to_dict(), study / "configuration.json")
    write_json({"point_fits": 24, "successful_fits": 24}, audit / "summary.json")
    write_json(
        {"requested_replays": 1, "verified_replays": int(condition != "replay")},
        replay / "summary.json",
    )
    (study / "observations.jsonl").write_text(
        "\n".join(json.dumps({**r, "size_bytes": 10000}) for r in probe.plan())
    )
    monkeypatch.setattr(
        gate.shutil,
        "disk_usage",
        lambda p: SimpleNamespace(free=7 * 1024**3 if condition == "space" else 20 * 1024**3),
    )
    output = tmp_path / "gate.json"
    if condition == "pass":
        result = gate.storage_preflight(
            c,
            probe_study=study,
            probe_audit=audit,
            probe_replay=replay,
            artifact_root=tmp_path,
            output_file=output,
        )
        assert result["status"] == "passed"
        assert result["planned_study_fits"] == 1200
        assert result["planned_replays"] == 50
    elif condition == "replay":
        with pytest.raises(ValueError, match="verified archived replay"):
            gate.storage_preflight(
                c,
                probe_study=study,
                probe_audit=audit,
                probe_replay=replay,
                artifact_root=tmp_path,
                output_file=output,
            )
        assert not output.exists()
    else:
        with pytest.raises(RuntimeError, match="Storage gate failed"):
            gate.storage_preflight(
                c,
                probe_study=study,
                probe_audit=audit,
                probe_replay=replay,
                artifact_root=tmp_path,
                output_file=output,
            )
        assert read_json(output)["status"] == "failed"
