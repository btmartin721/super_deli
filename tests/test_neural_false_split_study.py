"""Invented CLI lifecycle exercises actual neural refits and archival full parity."""

# ruff: noqa: E402

import json
import subprocess
import sys

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("sklearn")


def test_complete_installed_style_lifecycle_and_verified_continuation(tmp_path):
    from super_deli.benchmarking.neural_false_split_archive import bind
    from super_deli.benchmarking.neural_false_split_config import NeuralFalseSplitStudy
    from super_deli.benchmarking.neural_false_split_fixture import create
    from super_deli.benchmarking.neural_false_split_study import freeze

    source, root = tmp_path / "source", tmp_path / "study"
    create(source)
    root.mkdir()
    (root / "approved-plan.md").write_text("Invented engineering fixture only")
    bind(source, root, fixture=True)
    freeze(root, NeuralFalseSplitStudy(fixture=True))
    sequence = [
        ("run", "preflight", "preflight"),
        ("run", "preflight", "preflight"),
        ("audit", "preflight", "preflight"),
        ("finalize", "preflight", "preflight"),
        ("gate", "preflight", "preflight"),
        ("pause", "main", "batch_00"),
        ("run", "main", "batch_00"),
        ("resume", "main", "batch_00"),
        ("run", "main", "batch_00"),
        ("audit", "main", "batch_00"),
        ("run", "main", "extras"),
        ("audit", "main", "extras"),
        ("finalize", "main", "extras"),
        ("report", "main", "extras"),
    ]
    for i, (action, phase, group) in enumerate(sequence):
        cmd = [
            sys.executable,
            "-m",
            "super_deli",
            "benchmark-neural-false-splits",
            action,
            "--archive-root",
            str(source),
            "--output-dir",
            str(root),
            "--phase",
            phase,
            "--group",
            group,
        ]
        if action == "gate":
            cmd.extend(["--staging-bytes", "0"])
        p = subprocess.run(cmd, capture_output=True, text=True)
        (root / f"command_{i:02d}.log").write_text(p.stderr + "\n" + p.stdout)
        assert p.returncode == 0, (i, action, p.stdout[-2000:], p.stderr[-4000:])
        if i == 6:
            assert json.loads(p.stdout)["completed"] is False
    summary = json.loads((root / "summary.json").read_text())
    assert summary["observations"] == 2
    assert summary["counts"]["main"]["workflows"] == 26
    assert summary["audits"]["main"]["verified_optimizer_updates"] == 13000
    assert summary["audits"]["preflight"]["verified_optimizer_updates"] == 10000
    assert summary["primary_rates"] == 8 and summary["secondary_paired_comparisons"] == 4
    diagnostic = json.loads((root / "precision-diagnostic.json").read_text())
    assert diagnostic["excluded_from_fresh_rates"] is True
    assert all(
        r["original"]["requested"] == 2 and r["sensitivity"]["requested"] == 3
        for r in diagnostic["arms"]
    )
    from super_deli.benchmarking.neural_false_split_study import run_group

    changed = root / "main/gaussian_independent_0000/raw/morphology.tsv"
    changed.write_text("Deliberately changed completed evidence")
    with pytest.raises(ValueError, match="evidence changed"):
        run_group(root, source, "main", "batch_00")
    assert (root / "STOP.json").exists()
    assert changed.read_text() == "Deliberately changed completed evidence"


@pytest.mark.parametrize("failed_call", [1, 2])
def test_new_worker_withholds_failed_observed_or_null(tmp_path, monkeypatch, failed_call):
    from super_deli.benchmarking.neural_false_split_audit import audit_unit
    from super_deli.benchmarking.neural_false_split_config import NeuralFalseSplitStudy, tasks
    from super_deli.benchmarking.neural_false_split_report import assessment
    from super_deli.benchmarking.neural_false_split_run import run_unit
    from super_deli.benchmarking.neural_false_split_state import atomic_json, seal
    from super_deli.models import separate_gmm

    original = separate_gmm.select_gaussian_mixture
    calls = []

    def controlled(*args, **kwargs):
        calls.append(1)
        if len(calls) == failed_call:
            raise ValueError("Deliberate engineering-only unavailable mixture")
        return original(*args, **kwargs)

    monkeypatch.setattr(separate_gmm, "select_gaussian_mixture", controlled)
    atomic_json(tmp_path / "frozen.json", dict(fixture=True))
    (tmp_path / "main").mkdir()
    task = tasks(NeuralFalseSplitStudy(fixture=True), "main")[0]
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
    assert result["failed_points"] == 1
    assert assessment(tmp_path, task, "neural")["p_value"] is None
    assert not (tmp_path / "STOP.json").exists()
