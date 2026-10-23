"""Scientific diagnostic contrasts, source integrity, no-refit and failure checks."""

import hashlib
import importlib.util
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from super_deli.io.config import read_json, write_json
from super_deli.models.full_grouped_diagnostic_config import (
    ARMS,
    PARTITIONS,
    FullGroupedDiagnosticConfig,
    variants,
)
from super_deli.models.grouped_config import GroupedStudyConfig

ROOT = Path(__file__).resolve().parents[1]
SCIENTIFIC = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None, reason="Install clustering extra"
)


def model():
    return GroupedStudyConfig.from_dict(
        read_json(ROOT / "examples/full-groupmean-calibration/study.json")
    ).calibration.model


@pytest.mark.parametrize(
    "options",
    [
        {"random_seed": -1},
        {"n_datasets": True},
        {"small_n": 25},
        {"large_n": 64},
        {"schema_version": 2},
    ],
)
def test_invalid_config(options):
    with pytest.raises(ValueError):
        FullGroupedDiagnosticConfig("a" * 64, "b" * 64, "c" * 64, **options)


def test_grid_has_only_prespecified_changes():
    m = model()
    arms = variants(m)
    assert tuple(arms) == ARMS
    assert arms["baseline"] == m
    assert arms["pcs_2"] == replace(m, n_pcs=2)
    assert arms["pcs_3"] == replace(m, n_pcs=3)
    assert arms["starts_100"] == replace(m, n_init=100)
    assert arms["shared_4"] == replace(m, covariance_type="tied")
    assert arms["source_4"] == replace(
        m, views=(replace(m.views[0], pca_fit="source_rows"), m.views[1])
    )
    for bad in (replace(m, n_pcs=2), replace(m, covariance_type="tied"), arms["source_4"]):
        with pytest.raises(ValueError, match="baseline"):
            variants(bad)
    assert all(not a.n_bootstrap and a.max_k == 4 for a in arms.values())


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    spec = importlib.util.spec_from_file_location(
        "full_smoke", ROOT / "examples/full-grouped-diagnostics/smoke.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path_factory.mktemp("full-diagnostic")
    source, config = module.make_source(
        root, ROOT / "examples/full-groupmean-calibration/smoke.json"
    )
    from super_deli.benchmarking.full_grouped_diagnostics import run_full_grouped_diagnostics

    summary = run_full_grouped_diagnostics(
        config, source_artifacts=source, output_dir=root / "study"
    )
    return root, source, config, summary


@SCIENTIFIC
def test_complete_grid_and_read_only_audit(completed, monkeypatch):
    from sklearn.mixture import GaussianMixture

    from super_deli.benchmarking.full_grouped_diagnostic_analysis import (
        audit_full_grouped_diagnostics,
    )
    from super_deli.benchmarking.recovery_inputs import inventory

    root, source, config, summary = completed
    before = inventory(root / "study")
    assert summary["planned_points"] == summary["successful_points"] == 18
    assert summary["successful_truth_initialized"] == 9 and summary["replayed_archived"] == 1
    assert summary["bootstrap_or_calibration_fits"] == summary["failed_points"] == 0

    def no_fit(*args, **kwargs):
        pytest.fail("The auditor attempted a model refit")

    monkeypatch.setattr(GaussianMixture, "fit", no_fit)
    report = audit_full_grouped_diagnostics(root / "study", output_dir=root / "audit")
    assert report["archived_tests"] == report["archived_draws"] == 7
    assert len(report["paired"]) == 15
    assert inventory(root / "study") == before
    assert config.random_seed == 20261113 and source.exists()


@SCIENTIFIC
def test_pinned_source_and_retained_tail_tampering(completed, tmp_path):
    from super_deli.benchmarking.full_grouped_diagnostic_analysis import (
        audit_full_grouped_diagnostics,
    )
    from super_deli.benchmarking.full_grouped_diagnostics import source_check
    from super_deli.benchmarking.recovery_inputs import inventory

    root, source, config, _ = completed
    with pytest.raises(ValueError, match="fingerprint"):
        source_check(source, replace(config, source_study_sha256="0" * 64))
    destination = tmp_path / "corrupt"
    shutil.copytree(root / "study", destination)
    path = destination / "retained_tests/complementary/dataset_0000/null_replicates.json"
    data = read_json(path)
    data[0]["statistic"] += 1
    path.unlink()
    write_json(data, path)
    (destination / "files.json").unlink()
    write_json(inventory(destination), destination / "files.json")
    with pytest.raises(ValueError, match="pinned source"):
        audit_full_grouped_diagnostics(destination, output_dir=tmp_path / "audit")


@SCIENTIFIC
@pytest.mark.parametrize("failed_arm", ["starts_100", "baseline"])
def test_failed_points_retained_and_truth_withheld(completed, tmp_path, monkeypatch, failed_arm):
    from super_deli.benchmarking.full_grouped_diagnostics import _dataset
    from super_deli.workflows import integrative

    root, _, _, _ = completed
    original = root / "study/fresh_64/dataset_0000"
    target = tmp_path / "fresh_64/dataset_0000"
    target.mkdir(parents=True)
    shutil.copytree(original / "raw", target / "raw")
    shutil.copyfile(original / "truth.json", target / "truth.json")
    runner = integrative.run_integrative

    def fail(declaration, **kwargs):
        assert "truth" not in kwargs and "labels" not in kwargs
        if kwargs["output_dir"].parent.name == failed_arm:
            raise ValueError("Injected diagnostic failure")
        return runner(declaration, **kwargs)

    monkeypatch.setattr(integrative, "run_integrative", fail)
    points, oracles = _dataset((tmp_path, {"cohort": "fresh_64", "repetition": 0}, model()))
    assert len(points) == 6 and sum(r["status"] == "failed" for r in points) == 1
    failed = next(r for r in points if r["status"] == "failed")
    assert "statistic" not in failed and "metrics" not in failed
    assert len(oracles) == 3 and tuple(r["partition"] for r in oracles) == PARTITIONS
    assert all(
        r["status"] == ("unattempted" if failed_arm == "baseline" else "complete") for r in oracles
    )


@SCIENTIFIC
@pytest.mark.parametrize("name", PARTITIONS)
def test_truth_initialized_dense_likelihood_and_tampering(completed, tmp_path, name):
    import numpy as np

    from super_deli.benchmarking.full_grouped_oracle import audit_oracle

    root, _, _, _ = completed
    directory = root / "study/fresh_64/dataset_0000"
    truth = np.array(read_json(directory / "truth.json")["labels"])
    with np.load(directory / "baseline/run/mixture.npz", allow_pickle=False) as arrays:
        scores = arrays["scores"].copy()
    target = tmp_path / "oracle"
    shutil.copytree(directory / f"truth_{name}", target)
    record = read_json(target / "result.json")
    audit_oracle(target, truth, scores, name, model(), record)
    with pytest.raises(AssertionError):
        audit_oracle(target, truth, scores, name, model(), {**record, "bic": record["bic"] + 1})
    with np.load(target / "initialization.npz", allow_pickle=False) as a:
        arrays = {key: a[key].copy() for key in a.files}
    arrays["means"][0, 0] += 0.1
    np.savez(target / "initialization.npz", **arrays)
    with pytest.raises(AssertionError):
        audit_oracle(target, truth, scores, name, model(), record)


@SCIENTIFIC
def test_retained_zero_statistic_ties_are_upper_tail(completed):
    from super_deli.benchmarking.full_grouped_power import audit_tests

    root, _, _, _ = completed
    study = GroupedStudyConfig.from_dict(read_json(root / "study/source_study.json"))
    rows = audit_tests(root / "study/retained_tests", study)
    zero = [r for r in rows if r["observed_statistic"] == 0]
    assert zero and all(r["p_value"] == 1 for r in zero)
    assert all(r["p_value"] in {0.5, 1} for r in rows)


def test_historical_default_and_calibration_modules_unchanged():
    expected = {
        "models/integrative.py": (
            "c4d1973b3dfdac829c46aa606cf6fb5767f7efa969b792253ea5dca292e151ac"
        ),
        "models/integrative_config.py": (
            "87e71c99d21d26f8d37ad49210ff543d7890f5a64b3e2c2aad30251751a0f60c"
        ),
        "models/mixture.py": ("989598159cff9cf5524ad9c5113d0d27dc4f8f4b26076cc3488adc1cc4824892"),
        "models/grouped_config.py": (
            "7e88c5a3b20e89f26ee9291a63af77a9939d76896ae539a188a224b6f1d01c1e"
        ),
        "workflows/grouped_calibration.py": (
            "c90223a06490145cf2217e6310ac65e12ac3f25c1cbac8b9ec40e2944b671fd9"
        ),
    }
    for name, digest in expected.items():
        assert hashlib.sha256((ROOT / "src/super_deli" / name).read_bytes()).hexdigest() == digest
