"""Actual invented end-to-end fits, continuation, archival parity and failures."""

# ruff: noqa: E402
import json
import subprocess
import sys

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("sklearn")


def test_full_invented_lifecycle_and_tamper_protection(tmp_path):
    from super_deli.benchmarking.neural_power_archive import bind
    from super_deli.benchmarking.neural_power_config import NeuralPowerStudy
    from super_deli.benchmarking.neural_power_fixture import create
    from super_deli.benchmarking.neural_power_study import engine

    source, root = tmp_path / "source", tmp_path / "study"
    create(source)
    root.mkdir()
    (root / "approved-plan.md").write_text("Invented engineering fixture")
    bind(source, root, fixture=True)
    e = engine(root, source)
    e.freeze(NeuralPowerStudy(fixture=True))
    sequence = [
        ("run", "preflight"),
        ("run", "preflight"),
        ("audit", "preflight"),
        ("gate", "preflight"),
        ("pause", "main"),
        ("run", "main"),
        ("resume", "main"),
        ("run", "main"),
        ("audit", "main"),
        ("report", "main"),
    ]
    for index, (action, phase) in enumerate(sequence):
        args = [
            sys.executable,
            "-m",
            "super_deli",
            "benchmark-neural-power",
            action,
            "--archive-root",
            str(source),
            "--output-dir",
            str(root),
            "--phase",
            phase,
            "--workers",
            "2",
        ]
        if action == "gate":
            args += ["--staging-bytes", "0"]
        p = subprocess.run(args, capture_output=True, text=True)
        (root / f"command_{index:02d}.log").write_text(p.stderr + "\n" + p.stdout)
        assert p.returncode == 0, (action, p.stdout[-2000:], p.stderr[-4000:])
        if index == 5:
            assert json.loads(p.stdout)["completed"] is False
    result = json.loads((root / "summary.json").read_text())
    assert result["counts"]["main"]["workflows"] == 12
    assert result["audits"]["main"]["verified_optimizer_updates"] == 6000
    assert result["primary_rates"] == 6 and result["paired_comparisons"] == 3
    changed = next((root / "main").glob("four_*/raw/morphology.tsv"))
    changed.write_text("Deliberate invented tampering")
    with pytest.raises(ValueError, match="evidence changed"):
        e.run("main")
    assert (root / "STOP.json").exists() and changed.read_text() == "Deliberate invented tampering"


@pytest.mark.parametrize("failed_call", [1, 2])
def test_failed_observed_or_draw_withholds_pvalue(tmp_path, monkeypatch, failed_call):
    from super_deli.benchmarking.neural_false_split_report import assessment
    from super_deli.benchmarking.neural_false_split_state import atomic_json, seal
    from super_deli.benchmarking.neural_power_audit import audit_unit
    from super_deli.benchmarking.neural_power_config import NeuralPowerStudy, tasks
    from super_deli.benchmarking.neural_power_run import run_unit
    from super_deli.models import separate_gmm

    original = separate_gmm.select_gaussian_mixture
    calls = []

    def controlled(*args, **kwargs):
        calls.append(1)
        if len(calls) == failed_call:
            raise ValueError("Invented unavailable mixture")
        return original(*args, **kwargs)

    monkeypatch.setattr(separate_gmm, "select_gaussian_mixture", controlled)
    atomic_json(tmp_path / "frozen.json", dict(fixture=True))
    (tmp_path / "main").mkdir()
    task = tasks(NeuralPowerStudy(fixture=True), "main")[0]
    job = dict(
        kind="fresh",
        task=task,
        phase_root=str(tmp_path / "main"),
        archive_root=str(tmp_path / "unused"),
        stop_file=str(tmp_path / "STOP.json"),
    )
    result = run_unit(job)
    seal(tmp_path, "main", job, result)
    rows = audit_unit(tmp_path / "main" / task["key"], job, tmp_path, cache={})
    assert sum(r["neural_fits"] for r in rows) == result["neural_fits"]
    assert result["failed_points"] == 1 and assessment(tmp_path, task, "neural")["p_value"] is None
    assert not (tmp_path / "STOP.json").exists()


def test_interrupted_prefix_is_retained_with_unknown_counts(tmp_path, monkeypatch):
    import os
    import socket
    import time

    from super_deli.benchmarking import neural_false_split_study as shared
    from super_deli.benchmarking.neural_false_split_state import atomic_json
    from super_deli.benchmarking.neural_power_config import NeuralPowerStudy, tasks
    from super_deli.benchmarking.neural_power_study import engine

    config = NeuralPowerStudy(fixture=True)
    task = tasks(config, "preflight")[0]
    job = dict(kind="fresh", task=task)
    atomic_json(tmp_path / "frozen.json", dict(fixture=True))
    atomic_json(tmp_path / "replay-binding.json", dict(replays=[]))
    partial = tmp_path / "preflight" / task["key"] / "neural/calibration/summary.json"
    partial.parent.mkdir(parents=True)
    partial.write_text('{"incomplete":')
    atomic_json(
        tmp_path / "state/active" / ("preflight_" + task["key"] + ".json"),
        dict(
            key=task["key"],
            started=time.monotonic() - 1,
            elapsed=None,
            pid=os.getpid(),
            host=socket.gethostname(),
            status="running",
        ),
    )
    monkeypatch.setattr(shared, "process_alive", lambda _: False)
    shared.recover_interrupted(tmp_path, "preflight", job)
    value = engine(tmp_path, tmp_path / "unused").execution(config, "preflight")
    assert value["completed"] and value["unknown_counter_units"] == 1
    assert value["records"][0]["attempted"] is None
    assert partial.read_text() == '{"incomplete":'
