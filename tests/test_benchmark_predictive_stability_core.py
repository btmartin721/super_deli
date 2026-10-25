"""Dependency-free explicit scope, budgets and stable legacy defaults."""

from dataclasses import replace
from pathlib import Path

import pytest

from super_deli.io.config import read_json
from super_deli.models.predictive_bootstrap_config import PredictiveBootstrapConfig
from super_deli.models.predictive_bootstrap_study import PredictiveBootstrapStudy

ROOT = Path(__file__).resolve().parents[1]


def config(name="study.json"):
    return PredictiveBootstrapStudy.from_dict(
        read_json(ROOT / "examples/predictive-stability" / name)
    )


def test_frozen_design():
    c = config()
    assert PredictiveBootstrapStudy.from_dict(c.to_dict()) == c
    assert c.probe() == config("preflight.json")
    assert len(c.plan()) == 40
    assert sum(r["planned_fits"] for r in c.plan()) + 90 == 24090
    assert sum(r["planned_fits"] for r in c.probe().plan()) == 4800
    assert c.model.covariance_type == "tied" and c.model.n_bootstrap == 0
    assert c.model.n_init == 10 and c.model.random_seed == 71323

    def seeds(d):
        return [r[k] for r in d.plan() for k in ("data_seed", "split_seed", "bootstrap_seed")]

    assert len(seeds(c)) == len(set(seeds(c))) == 120
    assert set(seeds(c)).isdisjoint(seeds(c.probe()))
    assert c.max_artifact_bytes == 4 * 1024**3 and c.min_free_bytes == 8 * 1024**3
    assert sorted(c.plan(), key=str) == sorted(
        replace(c, cells=tuple(reversed(c.cells))).plan(), key=str
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("n_bootstrap", 0),
        ("n_bootstrap", True),
        ("n_bootstrap", 1.5),
        ("random_seed", -1),
        ("random_seed", True),
        ("schema_version", True),
        ("schema_version", 2),
    ],
)
def test_invalid_resampling(key, value):
    raw = {"n_bootstrap": 99, "random_seed": 321, "schema_version": 1, key: value}
    with pytest.raises(ValueError):
        PredictiveBootstrapConfig.from_dict(raw)


def test_explicit_fields_and_legacy_guards():
    with pytest.raises(ValueError):
        PredictiveBootstrapConfig.from_dict({"n_bootstrap": 99})
    with pytest.raises(ValueError):
        PredictiveBootstrapConfig.from_dict(
            dict(n_bootstrap=99, random_seed=2, schema_version=1, extra=3)
        )
    c = config()
    with pytest.raises(ValueError, match="stability"):
        replace(c.model, n_bootstrap=1)
    for model in (
        replace(c.model, covariance_type="full"),
        replace(c.model, target_level="population"),
    ):
        with pytest.raises(ValueError):
            PredictiveBootstrapConfig.validate_model(model, c.screening)
    with pytest.raises(ValueError):
        PredictiveBootstrapConfig.validate_model(c.model, replace(c.screening, n_folds=3))


def test_gate_required_before_main(tmp_path):
    from super_deli.benchmarking.predictive_bootstrap_study import run_study

    with pytest.raises(ValueError, match="preflight"):
        run_study(config(), output_dir=tmp_path / "no-run")
    assert not (tmp_path / "no-run").exists()


@pytest.mark.parametrize("condition", ["pass", "space", "cap", "incomplete"])
def test_audited_storage_gate(tmp_path, monkeypatch, condition):
    import json
    from types import SimpleNamespace

    import super_deli.benchmarking.predictive_bootstrap_cli as gate
    from super_deli.io.config import write_json

    c = config()
    if condition == "cap":
        c = replace(c, max_artifact_bytes=128 * 1024**2)
    probe = c.probe()
    source, audit = tmp_path / "probe", tmp_path / "audit"
    source.mkdir()
    audit.mkdir()
    write_json(probe.to_dict(), source / "configuration.json")
    write_json(
        {
            "audited_searches": 4800,
            "unattempted_draws": int(condition == "incomplete"),
            "failed_draws": 0,
        },
        audit / "summary.json",
    )
    rows = []
    for i, row in enumerate(probe.plan()):
        manifest = f"manifest_{i}.json"
        (source / manifest).write_text("{}")
        rows.append({**row, "size_bytes": 10000, "manifest": manifest})
    (source / "observations.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr(
        gate.shutil,
        "disk_usage",
        lambda p: SimpleNamespace(free=(7 if condition == "space" else 20) * 1024**3),
    )
    args = dict(
        probe_study=source,
        probe_audit=audit,
        artifact_root=tmp_path,
        output_file=tmp_path / "gate.json",
    )
    if condition == "pass":
        r = gate.storage_preflight(c, **args)
        assert r["status"] == "passed" and r["main_searches_including_replays"] == 24090
    elif condition == "incomplete":
        with pytest.raises(ValueError):
            gate.storage_preflight(c, **args)
        assert not (tmp_path / "gate.json").exists()
    else:
        with pytest.raises(RuntimeError):
            gate.storage_preflight(c, **args)
        assert read_json(tmp_path / "gate.json")["status"] == "failed"


def test_storage_never_descends_into_transient_workspaces(tmp_path, monkeypatch):
    import os

    from super_deli.benchmarking.predictive_study import retained_bytes

    for name in (".working", ".scratch"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "temporary").write_bytes(b"transient")
    (tmp_path / "retained").mkdir()
    (tmp_path / "retained/archive").write_bytes(b"saved")
    original = os.scandir

    def concurrent_scandir(path):
        if Path(path).name in {".working", ".scratch"}:
            raise AssertionError("Storage traversal entered a concurrently removed workspace")
        return original(path)

    monkeypatch.setattr(os, "scandir", concurrent_scandir)
    assert retained_bytes(tmp_path) == len(b"saved")


def test_storage_permission_errors_are_not_silently_undercounted(tmp_path, monkeypatch):
    import os

    from super_deli.benchmarking.predictive_study import retained_bytes

    (tmp_path / "retained").mkdir()
    original = os.scandir

    def denied(path):
        if Path(path).name == "retained":
            raise PermissionError("Cannot inspect retained evidence")
        return original(path)

    monkeypatch.setattr(os, "scandir", denied)
    with pytest.raises(PermissionError):
        retained_bytes(tmp_path)


def test_installed_smoke_retains_and_prints_child_diagnostics(tmp_path, capsys):
    import importlib.util
    import subprocess

    path = ROOT / "examples/predictive-stability/smoke.py"
    spec = importlib.util.spec_from_file_location("stability_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(subprocess.CalledProcessError):
        module.command(tmp_path, "bad", "not-a-super-deli-command")
    assert (tmp_path / "bad.stdout.log").exists()
    error = (tmp_path / "bad.stderr.log").read_text()
    assert "invalid choice" in error and "invalid choice" in capsys.readouterr().err
