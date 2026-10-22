"""Small paired study, failure retention, source preservation and no-refit audit."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

from super_deli.io.config import read_json, write_json
from super_deli.models.joint_config import JointStudyConfig
from super_deli.models.shared_config import SharedStudyConfig

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [
    pytest.mark.genomic,
    pytest.mark.skipif(
        importlib.util.find_spec("snpio") is None or importlib.util.find_spec("sklearn") is None,
        reason="Install scientific extras",
    ),
]


@pytest.fixture
def source(tmp_path):
    from super_deli.benchmarking.config import derived_seed
    from super_deli.benchmarking.joint_study import simulate_joint
    from super_deli.benchmarking.recovery_inputs import inventory, stage_generated
    from super_deli.benchmarking.shared_study import point_metrics
    from super_deli.core.provenance import fingerprint_file
    from super_deli.workflows.calibration_artifacts import fit_pipeline
    from super_deli.workflows.integrative import run_integrative

    study = JointStudyConfig.from_dict(read_json(ROOT / "examples/joint-calibration/smoke.json"))
    study = replace(
        study,
        calibration=replace(study.calibration, model=replace(study.calibration.model, max_k=4)),
    )
    root = tmp_path / "source"
    (root / "study/archived").mkdir(parents=True)
    write_json(study.to_dict(), root / "study/source_study.json")
    seed = derived_seed(study.random_seed, "complementary", 0, "observed_data")
    values, layout, labels = simulate_joint("complementary", 24, seed)
    base = root / "study/archived/dataset_0000"
    stage_generated(values, layout, labels, base)
    (base / "baseline").mkdir()
    from super_deli.io.config import load_config

    record = fit_pipeline(
        base / "baseline",
        load_config(base / "raw/dataset.json"),
        base / "raw",
        "point",
        model=study.calibration.model,
        runner=run_integrative,
    )
    record["metrics"] = point_metrics(base / "baseline/run", labels, True)
    write_json(record, base / "baseline/record.json")
    write_json({"scope": "minimal archived-baseline test fixture"}, root / "study/summary.json")
    write_json(inventory(root / "study"), root / "study/files.json")
    config = SharedStudyConfig(
        study.calibration.model,
        fingerprint_file(root / "study/files.json").sha256,
        fingerprint_file(root / "study/summary.json").sha256,
        fingerprint_file(root / "study/source_study.json").sha256,
        n_datasets=1,
        small_n=24,
        large_n=32,
        stability_datasets=1,
        stability_draws=2,
    )
    return root, config


def test_paired_study_independent_audit_and_failure_no_replacement(source, tmp_path, monkeypatch):
    from super_deli.benchmarking.recovery_inputs import inventory
    from super_deli.benchmarking.shared_analysis import audit_shared_study
    from super_deli.benchmarking.shared_study import run_shared_study
    from super_deli.models import mixture
    from super_deli.workflows import integrative

    original = integrative.run_integrative
    calls = []

    def runner(*args, **kwargs):
        calls.append(None)
        if len(calls) == 2:
            raise ValueError("Injected tied point failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(integrative, "run_integrative", runner)
    root, config = source
    before = inventory(root)
    result = run_shared_study(config, source_artifacts=root, output_dir=tmp_path / "study")
    assert len(calls) == 12 and result["successful_points"] == 11 and result["failed_points"] == 1
    assert result["planned_bootstrap"] == result["successful_bootstrap"] == 8
    assert inventory(root) == before
    monkeypatch.setattr(
        mixture, "select_gaussian_mixture", lambda *a, **kw: pytest.fail("Audit must not refit")
    )
    audited = audit_shared_study(tmp_path / "study", output_dir=tmp_path / "audit")
    assert audited["successful_points"] == 11 and audited["failed_points"] == 1
    assert audited["successful_bootstrap"] == 8 and len(audited["summaries"]) == 12
    with pytest.raises(FileExistsError):
        run_shared_study(config, source_artifacts=root, output_dir=tmp_path / "study")
    (tmp_path / "study/fresh_64/dataset_0000/tied/run/bootstrap_replicates.json").write_text("[]\n")
    with pytest.raises(ValueError, match="inventory"):
        audit_shared_study(tmp_path / "study", output_dir=tmp_path / "corrupt")


def test_failed_bootstrap_withholds_support_and_preserves_bounds(source, tmp_path, monkeypatch):
    from super_deli.benchmarking.shared_analysis import audit_shared_study
    from super_deli.benchmarking.shared_study import run_shared_study
    from super_deli.models import integrative

    original = integrative._fit_once
    failed = []

    def once(inputs, units, config, seed, draw):
        if config.covariance_type == "tied" and len(set(draw)) < len(units) and not failed:
            failed.append(True)
            raise ValueError("Injected shared bootstrap failure")
        return original(inputs, units, config, seed, draw)

    monkeypatch.setattr(integrative, "_fit_once", once)
    root, config = source
    result = run_shared_study(config, source_artifacts=root, output_dir=tmp_path / "study")
    assert result["failed_points"] == 0 and result["failed_bootstrap"] == 1
    assert result["attempted_bootstrap"] == 8 and result["unattempted_bootstrap"] == 0
    audit = audit_shared_study(tmp_path / "study", output_dir=tmp_path / "audit")
    assert audit["failed_bootstrap"] == 1
    clusters = read_json(tmp_path / "study/fresh_64/dataset_0000/tied/run/clusters.json")
    assert all(r["frequency"] is None and r["failure_upper"] > r["failure_lower"] for r in clusters)


def test_corrupt_source_rejected_before_generating_fresh_data(source, tmp_path):
    from super_deli.benchmarking.shared_study import run_shared_study

    root, config = source
    (root / "study/archived/dataset_0000/truth.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        run_shared_study(config, source_artifacts=root, output_dir=tmp_path / "study")
    assert not (tmp_path / "study").exists()
