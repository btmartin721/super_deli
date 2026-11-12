# ruff: noqa: E402
# Check optional libraries before importing numerical diagnostic modules.
"""Independent known-law checks, retained failures and sealed engineering lifecycle."""

import json
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")
from scipy.stats import norm

from super_deli.benchmarking.neural_false_split_state import verify_seal
from super_deli.benchmarking.weak_signal_audit import audit_reference
from super_deli.benchmarking.weak_signal_config import WeakSignalStudy
from super_deli.benchmarking.weak_signal_fixture import create
from super_deli.benchmarking.weak_signal_reference import (
    classify,
    expected_accuracy,
    latent_law,
    reference_result,
)
from super_deli.benchmarking.weak_signal_report import rate
from super_deli.benchmarking.weak_signal_source import bind, check_inputs
from super_deli.benchmarking.weak_signal_study import engine


@pytest.mark.parametrize("strength", [0.5, 1.0, 1.5])
def test_complementary_law_and_analytic_error(strength):
    mean, covariance = latent_law(dict(case="four_complementary", signal_multiplier=strength))
    a = 3 * strength
    np.testing.assert_array_equal(
        mean, [[-a, 0, 0, -a], [-a, 0, 0, a], [a, 0, 0, -a], [a, 0, 0, a]]
    )
    assert expected_accuracy(mean, covariance) == pytest.approx(norm.cdf(a) ** 2)
    p, labels = classify(mean, mean, covariance)
    np.testing.assert_array_equal(labels, np.arange(4))
    np.testing.assert_allclose(p.sum(1), 1)
    # A model-implied classification reference does not assert a finite-sample ARI bound.
    assert reference_result(mean, np.arange(4), mean, covariance)[0]["recovered"]


def test_invertible_scaling_preserves_reference():
    mean, cov = latent_law(dict(case="four_complementary", signal_multiplier=0.5))
    x = np.array([[1, 0.5, -1, -0.1], [-2, 0, 0.1, 0.5]])
    transform = np.diag([2, 0.5, 3, 0.2])
    offset = np.array([7.0, -1, 2, 4])
    a, _ = classify(x, mean, cov)
    b, _ = classify(
        x @ transform.T + offset, mean @ transform.T + offset, transform @ cov @ transform.T
    )
    np.testing.assert_allclose(a, b, atol=1e-12)
    assert expected_accuracy(mean, cov) == pytest.approx(
        expected_accuracy(mean @ transform.T, transform @ cov @ transform.T)
    )


@pytest.mark.parametrize("field", ["means", "covariance", "membership", "labels"])
def test_independent_audit_rejects_changed_reference(field):
    mean, cov = latent_law(dict(case="four_complementary", signal_multiplier=1.0))
    row, saved = reference_result(mean, np.arange(4), mean, cov)
    audit_reference(mean, np.arange(4), saved, row, (mean, cov))
    saved[field] = saved[field].copy()
    saved[field].flat[0] += 0.1 if field != "labels" else 1
    with pytest.raises((AssertionError, ValueError)):
        audit_reference(mean, np.arange(4), saved, row, (mean, cov))


def test_failure_denominators():
    r = rate([True, False, None])
    assert (r["count"], r["requested"], r["unavailable"]) == (1, 3, 1)
    assert r["failure_bounds"] == [1 / 3, 2 / 3]
    assert r["descriptive_95_interval"][0] < 1 / 3 < r["descriptive_95_interval"][1]


@pytest.fixture(scope="module")
def invented(tmp_path_factory):
    base = tmp_path_factory.mktemp("weak-source")
    create(base / "source")
    return base / "source"


def prepare(source, root):
    root.mkdir()
    (root / "approved-plan.md").write_text("Invented engineering protocol")
    bind(source, root, fixture=True)
    study = engine(root, source)
    study.freeze(WeakSignalStudy(fixture=True))
    return study


def test_full_sealed_fixture_without_optional_imports(invented, tmp_path, monkeypatch):
    # Full scientific-suite collection may already have imported SNPio. The
    # diagnostic requires a clean process, just like its CLI/installed example.
    import types

    monkeypatch.setitem(sys.modules, "snpio", types.ModuleType("snpio"))
    root = tmp_path / "study"
    code = """import sys
from pathlib import Path
from super_deli.benchmarking.coordinate_fit import frozen_only
from super_deli.benchmarking.weak_signal_config import WeakSignalStudy
from super_deli.benchmarking.weak_signal_source import bind
from super_deli.benchmarking.weak_signal_study import engine, gate
from super_deli.benchmarking.weak_signal_report import report
source, root = map(Path, sys.argv[1:])
with frozen_only():
 root.mkdir(); (root/'approved-plan.md').write_text('Invented engineering protocol')
 bind(source, root, fixture=True)
 study = engine(root, source); study.freeze(WeakSignalStudy(fixture=True))
 for phase in ('preflight', 'main'):
  if phase == 'main': assert gate(root, source, staging_bytes=0)['passed']
  executed = study.run(phase, workers=1)
  assert executed['attempted'] == 28 and executed['neural_fits'] == 0
  assert study.audit(phase)['passed']
  assert study.run(phase, workers=1)['completed']
 result = report(root, source)
 assert result['observations'] == 8 and result['reference_evaluations'] == 24
 assert result['component_fits'] == 28 and result['archived_candidate_densities'] == 64
assert 'torch' not in sys.modules and 'snpio' not in sys.modules
print('isolated lifecycle passed')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(invented), str(root)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "isolated lifecycle passed" in result.stdout


def test_modified_extract_rejected(invented, tmp_path):
    study = prepare(invented, tmp_path / "study")
    path = next((study.root / "inputs").glob("*/task.json"))
    path.write_text("{}")
    with pytest.raises(ValueError, match="Immutable source extracts"):
        check_inputs(study.root)
    with pytest.raises(ValueError):
        study.check()


def test_original_stop_prohibits_run(invented, tmp_path):
    study = prepare(invented, tmp_path / "study")
    (study.root / "STOP.json").write_text(json.dumps(dict(reason="Retained failure")))
    with pytest.raises(ValueError, match="amendment"):
        study.run("preflight")


def test_guard_rejects_modified_unit(invented, tmp_path):
    study = prepare(invented, tmp_path / "study")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "super_deli",
            "diagnose-weak-signal",
            "run",
            "--archive-root",
            str(invented),
            "--output-dir",
            str(study.root),
            "--phase",
            "preflight",
            "--workers",
            "1",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    job = study.jobs(WeakSignalStudy(fixture=True), "preflight")[0]
    path = study.root / "preflight" / job["task"]["key"] / "analysis.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="evidence changed"):
        verify_seal(study.root, "preflight", job)


@pytest.mark.parametrize("strength", [0.5, 1.0, 1.5])
def test_measurement_noise_lowers_known_parameter_accuracy(strength):
    from super_deli.benchmarking.weak_signal_reference import loading

    means, covariance = latent_law(dict(case="four_complementary", signal_multiplier=strength))
    matrix = loading()
    raw_covariance = matrix @ covariance @ matrix.T + 0.0225 * np.eye(12)
    raw_accuracy = expected_accuracy(means @ matrix.T, raw_covariance)
    assert 0.25 < raw_accuracy < expected_accuracy(means, covariance)


def test_failed_truth_start_is_retained_without_replacement(monkeypatch, tmp_path):
    from super_deli.benchmarking import weak_signal_fit
    from super_deli.benchmarking.weak_signal_audit import audit_truth

    calls = []

    def fail(self, values):
        calls.append(values.copy())
        raise ValueError("Invented optimizer failure")

    monkeypatch.setattr(weak_signal_fit.GaussianMixture, "fit", fail)
    truth = np.repeat(np.arange(4), 8)
    x = np.random.default_rng(45).normal(size=(32, 4))
    out = tmp_path / "retained"
    row = weak_signal_fit.truth_fit(x, truth, out)
    assert row["status"] == "failed" and row["attempted"] == 1 and len(calls) == 1
    assert "Invented optimizer failure" in row["error"]
    assert (out / "initialization.npz").exists() and (out / "fit.npz").exists()
    assert audit_truth(out, x, truth) == row
