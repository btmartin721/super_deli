"""All-case end-to-end calibration, no-refit audit and corruption rejection."""

# ruff: noqa: E402
import importlib.util
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from super_deli.benchmarking.joint_analysis import audit_joint_study
from super_deli.benchmarking.joint_study import run_joint_study
from super_deli.cli import main
from super_deli.io.config import read_json
from super_deli.models.joint_config import CASES, JointStudyConfig

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [
    pytest.mark.genomic,
    pytest.mark.skipif(importlib.util.find_spec("snpio") is None, reason="Install genomics extra"),
]


@pytest.mark.parametrize("family", ["full", "tied"])
def test_all_cases_direct_cli_and_independent_audit(tmp_path, monkeypatch, family):
    study = JointStudyConfig.from_dict(read_json(ROOT / "examples/joint-calibration/smoke.json"))
    study = replace(
        study,
        calibration=replace(
            study.calibration, model=replace(study.calibration.model, covariance_type=family)
        ),
    )
    root = tmp_path / "study"
    root.mkdir()
    for case in CASES:
        result = run_joint_study(study, case=case, output_dir=root / case)
        assert result["successful_fits"] == 3
    base = root / "linear_association/dataset_0000"
    assert (
        main(
            [
                "calibrate-joint-integrative",
                str(base / "observed/dataset.json"),
                "--calibration-config",
                str(base / "calibration/configuration.json"),
                "--output-dir",
                str(tmp_path / "direct"),
            ]
        )
        == 0
    )
    direct = read_json(tmp_path / "direct/summary.json")
    evidence = read_json(tmp_path / "direct/evidence.json")
    assert direct["planned_fits"] == direct["successful_fits"] == 3
    assert ("prespecified shared covariance" in evidence["method"]) == (family == "tied")
    assert ("fixed before analysis" in direct["scope"]) == (family == "tied")
    for case in CASES:
        configuration = read_json(root / case / "dataset_0000/calibration/configuration.json")
        assert configuration["model"].get("covariance_type", "full") == family
    from super_deli.benchmarking.pipeline_analysis import audit_pipeline_point
    from super_deli.models import mixture
    from super_deli.models.joint_config import JointCalibrationConfig

    calibration = base / "calibration"
    policy = JointCalibrationConfig.from_dict(read_json(calibration / "configuration.json"))
    other = replace(
        policy, model=replace(policy.model, covariance_type="full" if family == "tied" else "tied")
    )
    with ZipFile(calibration / "pipeline_artifacts.zip") as archive:
        with pytest.raises(ValueError, match="configuration mismatch"):
            audit_pipeline_point(archive, read_json(calibration / "observed.json"), other)

    monkeypatch.setattr(
        mixture, "select_gaussian_mixture", lambda *a, **kw: pytest.fail("Audit must not refit")
    )
    audited = audit_joint_study(root, output_dir=tmp_path / "audit")
    assert audited["verified_successful_pipelines"] == 15 and audited["datasets"] == 5
    path = base / "observed/phenotypic/view_000.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="raw input"):
        audit_joint_study(root, output_dir=tmp_path / "corrupt")


@pytest.mark.parametrize("family", ["full", "tied"])
@pytest.mark.parametrize("failure_call", [1, 2])
def test_failed_joint_fit_withholds_p_without_replacement(
    tmp_path, monkeypatch, failure_call, family
):
    from super_deli.benchmarking.joint_study import simulate_joint
    from super_deli.models.joint_null import write_raw_joint
    from super_deli.workflows import integrative
    from super_deli.workflows.joint_calibration import run_joint_calibration

    c = JointStudyConfig.from_dict(
        read_json(ROOT / "examples/joint-calibration/smoke.json")
    ).calibration
    c = replace(c, model=replace(c.model, covariance_type=family))
    values, layout, _ = simulate_joint("linear_association", 24, 27)
    dataset = write_raw_joint(values, layout, tmp_path / "raw")
    original = integrative.run_integrative
    calls = []

    def run(*args, **kwargs):
        calls.append(None)
        if len(calls) == failure_call:
            raise ValueError("injected joint failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(integrative, "run_integrative", run)
    result = run_joint_calibration(
        dataset,
        config=replace(c, n_simulations=2),
        base_dir=tmp_path / "raw",
        output_dir=tmp_path / "fit",
    )
    assert result["inference"]["p_value"] is None and result["failed_fits"] == 1
    assert len(calls) == (1 if failure_call == 1 else 3)
    assert result["unattempted_fits"] == (2 if failure_call == 1 else 0)


@pytest.mark.parametrize("family", ["full", "tied"])
@pytest.mark.parametrize("kind", ["morphological", "ecological"])
def test_single_phenotype_and_id_matching(tmp_path, kind, family):
    from super_deli.benchmarking.joint_study import simulate_joint
    from super_deli.models.joint_null import write_raw_joint
    from super_deli.workflows.joint_calibration import run_joint_calibration

    c = JointStudyConfig.from_dict(
        read_json(ROOT / "examples/joint-calibration/smoke.json")
    ).calibration
    c = replace(
        c,
        n_simulations=1,
        model=replace(
            c.model,
            covariance_type=family,
            views=tuple(v for v in c.model.views if v.kind in {"genomic", kind}),
        ),
    )
    values, layout, _ = simulate_joint("linear_association", 24, 29)
    dataset = write_raw_joint(values, layout, tmp_path / "raw")
    dataset = replace(
        dataset, modalities=tuple(m for m in dataset.modalities if m.kind in {"genomic", kind})
    )
    result = run_joint_calibration(
        dataset, config=c, base_dir=tmp_path / "raw", output_dir=tmp_path / "fit"
    )
    assert result["n_raw_features"] == 132 and result["successful_fits"] == 2
