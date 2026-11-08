"""Invented full lifecycle and independent raw-law references."""

# ruff: noqa: E402

import json
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("sklearn")

from super_deli.benchmarking.neural_calibration_config import CASES, NeuralCalibrationStudy, tasks
from super_deli.benchmarking.neural_calibration_data import generate, reference


@pytest.mark.parametrize("case", CASES)
def test_independent_raw_reference(case):
    t = {**tasks(NeuralCalibrationStudy(fixture=True), "preflight")[0], "case": case}
    a, b = generate(t), reference(t)
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_allclose(a[key], b[key], atol=1e-12)


def test_complete_cli_lifecycle(tmp_path):
    from super_deli.benchmarking.neural_calibration_archive import bind
    from super_deli.benchmarking.neural_calibration_fixture import create
    from super_deli.benchmarking.neural_calibration_study import freeze

    source = tmp_path / "source"
    create(source)
    root = tmp_path / "study"
    root.mkdir()
    (root / "approved-plan.md").write_text("Invented engineering fixture only")
    bind(source, root, fixture=True)
    freeze(root, NeuralCalibrationStudy(fixture=True))
    for action in ("preflight", "audit-preflight", "gate", "main", "audit-main", "report"):
        command = [
            sys.executable,
            "-m",
            "super_deli",
            "benchmark-neural-calibration",
            action,
            "--archive-root",
            str(source),
            "--output-dir",
            str(root),
        ]
        if action == "gate":
            command.extend(["--staging-bytes", "0"])
        p = subprocess.run(command, capture_output=True, text=True)
        (root / (action + ".log")).write_text(p.stderr + "\n" + p.stdout)
        assert p.returncode == 0, (action, p.stdout[-2000:], p.stderr[-4000:])
    summary = json.loads((root / "summary.json").read_text())
    assert summary["observations"] == 1
    assert summary["audits"]["preflight"]["verified_optimizer_updates"] == 3000
    assert summary["audits"]["main"]["verified_optimizer_updates"] == 4000
    assert summary["counts"]["main"]["workflows"] == 8


@pytest.mark.parametrize("failed_call", [1, 2])
def test_independent_audit_retains_failed_observed_or_null(monkeypatch, tmp_path, failed_call):
    from super_deli.benchmarking.neural_calibration_audit import audit_dataset
    from super_deli.benchmarking.neural_calibration_run import run_dataset
    from super_deli.io.config import read_json
    from super_deli.models import separate_gmm

    original = separate_gmm.select_gaussian_mixture
    calls = []

    def selected(*args, **kwargs):
        calls.append(1)
        if len(calls) == failed_call:
            raise ValueError("Deliberate unavailable point search")
        return original(*args, **kwargs)

    monkeypatch.setattr(separate_gmm, "select_gaussian_mixture", selected)
    t = tasks(NeuralCalibrationStudy(fixture=True), "preflight")[0]
    result = run_dataset(
        dict(task=t, phase_root=str(tmp_path), stop_file=str(tmp_path / "STOP.json"))
    )
    audited = audit_dataset(tmp_path / t["key"], t, cache={})
    assert sum(a["neural_fits"] for a in audited) == result["neural_fits"]
    result = read_json(tmp_path / t["key"] / "neural/calibration/summary.json")
    assert result["inference"]["p_value"] is None
    assert not (tmp_path / "STOP.json").exists()
