"""Scenario partitioning, independent raw-log audit and Monte Carlo reporting."""

# ruff: noqa: E402

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from super_deli.benchmarking.config import Arm, BenchmarkConfig
from super_deli.benchmarking.study_analysis import (
    audit_bootstrap,
    audit_study,
    rate_summary,
    summarize_study,
    write_analysis,
)
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "study_execution", ROOT / "scripts/run_integrative_study.py"
)
execution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(execution)


def configuration():
    original = BenchmarkConfig.from_dict(read_json(ROOT / "examples/benchmarking/smoke.json"))
    return replace(
        original,
        n_datasets=1,
        scenarios=(original.scenarios[6], replace(original.scenarios[-1], view_missing_rate=0.2)),
        arms=(original.arms[2], Arm("traits_environment", ("morphological", "ecological"))),
        model=replace(original.model, n_bootstrap=3, n_init=2),
    )


@pytest.fixture
def study(tmp_path):
    path = tmp_path / "study"
    execution.run_study(configuration(), output_dir=path, python=sys.executable, workers=2)
    return path


def test_partition_changes_only_scenario_list():
    config = configuration()
    for scenario, shard in zip(
        config.scenarios, execution.scenario_configs(config).values(), strict=True
    ):
        assert BenchmarkConfig.from_dict(shard) == replace(config, scenarios=(scenario,))


def test_complete_study_audit_recomputes_without_refitting(study, tmp_path, monkeypatch):
    from super_deli.models import integrative

    def forbidden(*args, **kwargs):
        raise AssertionError("Analysis must not refit a model")

    monkeypatch.setattr(integrative, "fit_integrative", forbidden)
    config, records, audit = audit_study(study)
    assert len(records) == 4 and audit["bootstrap"] == {
        "requested": 12,
        "successful": 12,
        "failed": 0,
    }
    assert config == configuration()
    output = tmp_path / "analysis"
    summary = write_analysis(study, output_dir=output)
    assert summary["point_failed"] == 0
    assert all(g["rates"]["point_success"]["wilson_95"][0] > 0 for g in summary["groups"])
    assert (output / "datasets.csv").is_file() and (output / "files.jsonl").is_file()
    before = (output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        write_analysis(study, output_dir=output)
    assert (output / "summary.json").read_bytes() == before
    with pytest.raises(ValueError, match="outside"):
        write_analysis(study, output_dir=study / "analysis")


@pytest.mark.parametrize("part", ["draw", "counts", "truth", "cohort", "completion"])
def test_tampered_artifacts_are_rejected(study, part):
    scenario = configuration().scenarios[0].name
    run = study / scenario / "runs" / f"{scenario}_0000" / configuration().arms[0].name
    if part == "draw":
        path = run / "bootstrap_replicates.json"
        records = read_json(path)
        records[0]["training_indices"][0] = -1
        path.write_text(json.dumps(records))
    elif part == "counts":
        path = run / "stability.npz"
        with np.load(path, allow_pickle=False) as arrays:
            changed = dict(arrays)
        changed["coassignment_counts"][0, 1] += 1
        np.savez(path, **changed)
    elif part == "truth":
        path = run.parent / "simulation/truth.json"
        path.write_text(path.read_text() + "\n")
    elif part == "cohort":
        path = run / "summary.json"
        value = read_json(path)
        value["unit_ids"].reverse()
        path.write_text(json.dumps(value))
    else:
        path = study / "execution-outcomes.json"
        value = read_json(path)
        value["status"] = "failed"
        path.write_text(json.dumps(value))
    with pytest.raises((ValueError, AssertionError)):
        audit_study(study)


def test_duplicate_run_identity_is_rejected_even_with_updated_hashes(study):
    scenario = configuration().scenarios[0].name
    shard = study / scenario / "runs"
    ledger = shard / "runs.jsonl"
    lines = ledger.read_text().splitlines()
    ledger.write_text("\n".join([lines[0], lines[0]]) + "\n")
    path = shard / "summary.json"
    summary = read_json(path)
    summary["runs_sha256"] = fingerprint_file(ledger).sha256
    path.write_text(json.dumps(summary))
    path = study / "execution-outcomes.json"
    outcomes = read_json(path)
    outcomes["outcomes"][0]["summary_sha256"] = fingerprint_file(shard / "summary.json").sha256
    path.write_text(json.dumps(outcomes))
    with pytest.raises(ValueError, match="duplicate"):
        audit_study(study)


def test_wilson_intervals_retain_boundary_uncertainty():
    zero, one = rate_summary([0] * 100, 100), rate_summary([1] * 100, 100)
    assert zero["wilson_95"][1] == pytest.approx(0.0369934982)
    assert one["wilson_95"][0] == pytest.approx(0.9630065018)
    assert zero["mcse"] == one["mcse"] == 0
    half = rate_summary([0, 1, None], 3)
    assert half["mean"] == 0.5 and half["n_available"] == 2 and half["n_missing"] == 1
    assert rate_summary([None], 1)["wilson_95"] is None
    with pytest.raises(ValueError):
        rate_summary([0.5], 1)


def test_failed_inner_draw_is_audited_and_wrong_cluster_event_is_unknown(tmp_path, monkeypatch):
    from super_deli.benchmarking.runner import run_benchmark
    from super_deli.models import integrative

    original, calls = integrative._fit_once, 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("deliberate inner draw failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(integrative, "_fit_once", fail_once)
    config = configuration()
    config = replace(config, scenarios=config.scenarios[:1], arms=config.arms[:1])
    root = tmp_path / "runs"
    run_benchmark(config, output_dir=root)
    record = json.loads((root / "runs.jsonl").read_text().splitlines()[0])
    counts = audit_bootstrap(root / record["run_path"], record["analysis_seed"], 3)
    assert counts["successful"] == 2 and counts["failed"] == 1
    assert counts["failure_reasons"] == {"deliberate inner draw failure": 1}
    record["bootstrap_failure_reasons"] = counts["failure_reasons"]
    summary = summarize_study(config, [record], {})
    event = summary["groups"][0]["rates"]["any_stable_wrong_cluster"]
    assert event["mean"] is None and event["n_missing"] == 1
    assert summary["groups"][0]["metrics"]["coassignment_squared_error"]["mean"] is None
