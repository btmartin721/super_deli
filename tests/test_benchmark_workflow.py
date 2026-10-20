"""Installed CLI, real SNPio, shared cohorts/streams and auditable study failures."""

# ruff: noqa: E402

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from super_deli.benchmarking.config import BenchmarkConfig
from super_deli.benchmarking.runner import run_benchmark
from super_deli.io.config import read_json

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/benchmarking/smoke.json"


def config(*, genomic=False):
    original = BenchmarkConfig.from_dict(json.loads(EXAMPLE.read_text()))
    arms = (original.arms[0], original.arms[2]) if genomic else (original.arms[2],)
    return replace(
        original,
        arms=arms,
        scenarios=(original.scenarios[6],),
        n_datasets=1,
        model=replace(original.model, n_bootstrap=3, n_init=2),
    )


def ledger(path):
    return [json.loads(line) for line in (path / "runs.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("grouped", [False, True])
def test_real_four_view_study_common_cohort_and_bootstrap_streams(tmp_path, grouped):
    pytest.importorskip("snpio")
    options = config(genomic=True)
    scenario = replace(
        options.scenarios[0], linkage="grouped" if grouped else "matched", unequal_groups=grouped
    )
    options = replace(options, scenarios=(scenario,))
    output = tmp_path / "study"
    summary = run_benchmark(options, output_dir=output)
    assert summary["attempted"] == 2 and summary["point_failed"] == 0
    records = ledger(output)
    assert len({r["simulation_seed"] for r in records}) == 1
    assert len({r["analysis_seed"] for r in records}) == 1
    paths = [output / r["run_path"] for r in records]
    summaries = [read_json(p / "summary.json") for p in paths]
    assert summaries[0]["unit_ids"] == summaries[1]["unit_ids"]
    assert len(summaries[0]["unit_ids"]) < options.n_units
    draws = [read_json(p / "bootstrap_replicates.json") for p in paths]
    for left, right in zip(*draws, strict=True):
        assert left["training_indices"] == right["training_indices"]
        assert left["draw_seed"] == right["draw_seed"]
    assert all(not list(p.rglob("truth.json")) for p in paths)
    assert summaries[0]["level"] == ("locality" if grouped else "specimen")
    # SNPio was actually used, including absent optional popmap.
    assert (paths[0] / "view_000/ingestion").is_dir()


def test_non_genomic_cli_reproducibility_and_exclusive_outputs(tmp_path):
    options = config()
    declaration = tmp_path / "benchmark.json"
    declaration.write_text(json.dumps(options.to_dict()))
    before = declaration.read_bytes(), declaration.stat().st_mtime_ns
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    first, second = tmp_path / "first", tmp_path / "second"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "super_deli",
            "benchmark-integrative",
            str(declaration),
            "--output-dir",
            str(first),
        ],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )
    summary = json.loads(result.stdout)
    assert summary["status"] == "complete" and summary["point_failed"] == 0
    run_benchmark(options, output_dir=second)
    a, b = ledger(first)[0], ledger(second)[0]
    assert a["evaluation"] == b["evaluation"]
    np.testing.assert_array_equal(
        np.load(first / a["run_path"] / "mixture.npz")["scores"],
        np.load(second / b["run_path"] / "mixture.npz")["scores"],
    )
    assert (declaration.read_bytes(), declaration.stat().st_mtime_ns) == before
    assert not list(cwd.iterdir())
    original = (first / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_benchmark(options, output_dir=first)
    assert (first / "summary.json").read_bytes() == original


def test_point_failure_is_recorded_and_remaining_runs_continue(tmp_path, monkeypatch):
    from super_deli.workflows import integrative

    original, calls = integrative.run_integrative, 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("deliberate numerical failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(integrative, "run_integrative", fail_once)
    options = replace(config(), n_datasets=2)
    output = tmp_path / "study"
    summary = run_benchmark(options, output_dir=output)
    assert summary["status"] == "complete" and summary["point_failed"] == 1
    assert [r["status"] for r in ledger(output)] == ["failed", "complete"]
    assert summary["groups"][0]["metrics"]["ari"]["n_missing"] == 1
    assert summary["groups"][0]["bootstrap_requested_including_failed_points"] == 6


def test_scoring_error_aborts_study_instead_of_becoming_model_failure(tmp_path, monkeypatch):
    from super_deli.benchmarking import metrics

    def fail(*args):
        raise ValueError("deliberate scorer error")

    monkeypatch.setattr(metrics, "score_run", fail)
    output = tmp_path / "study"
    with pytest.raises(ValueError, match="scorer"):
        run_benchmark(config(), output_dir=output)
    assert (output / "failure.json").is_file() and not (output / "summary.json").exists()
