"""Optional covariance family, compatibility and significance boundaries."""

from dataclasses import replace
from pathlib import Path

import pytest

from super_deli.io.config import read_json
from super_deli.models.integrative_config import IntegrativeConfig
from super_deli.models.joint_config import JointStudyConfig
from super_deli.models.mixture_parameters import mixture_parameter_count

ROOT = Path(__file__).resolve().parents[1]


def baseline():
    return JointStudyConfig.from_dict(read_json(ROOT / "examples/joint-calibration/smoke.json"))


def test_explicit_full_and_nested_defaults_preserve_old_configuration_bytes():
    original = baseline()
    model = original.calibration.model
    assert "covariance_type" not in model.to_dict()
    explicit = IntegrativeConfig.from_dict({**model.to_dict(), "covariance_type": "full"})
    assert explicit.to_dict() == model.to_dict()
    assert (
        replace(original, calibration=replace(original.calibration, model=explicit)).to_dict()
        == original.to_dict()
    )
    tied = replace(model, covariance_type="tied")
    assert tied.to_dict()["covariance_type"] == "tied"
    assert IntegrativeConfig.from_dict(tied.to_dict()) == tied
    assert mixture_parameter_count(4, 2, "tied") == 14
    assert mixture_parameter_count(4, 2, "full") == 23


@pytest.mark.parametrize("value", [None, True, "diag", "spherical", [], "TIED"])
def test_unsupported_covariance_rejected(value):
    with pytest.raises(ValueError, match="covariance_type"):
        replace(baseline().calibration.model, covariance_type=value)


@pytest.mark.parametrize(
    "name,kind",
    [
        ("pipeline", "pipeline"),
        ("empirical", "empirical"),
        ("grouped", "grouped"),
    ],
)
def test_raw_significance_rejects_unvalidated_covariance(name, kind):
    from super_deli.models.empirical_config import EmpiricalStudyConfig
    from super_deli.models.grouped_config import GroupedStudyConfig
    from super_deli.models.pipeline_config import PipelineStudyConfig

    classes = {
        "pipeline": PipelineStudyConfig,
        "empirical": EmpiricalStudyConfig,
        "grouped": GroupedStudyConfig,
        "joint": JointStudyConfig,
    }
    study = classes[kind].from_dict(read_json(ROOT / f"examples/{name}-calibration/smoke.json"))
    with pytest.raises(ValueError, match="not yet supported"):
        replace(study.calibration, model=replace(study.calibration.model, covariance_type="tied"))


def test_direct_score_significance_rejects_unvalidated_covariance():
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    from super_deli.models.calibration_config import GaussianCalibrationConfig
    from super_deli.models.gaussian_calibration import calibrate_gaussian_scores

    with pytest.raises(ValueError, match="not yet supported"):
        calibrate_gaussian_scores(
            np.random.default_rng(1).normal(size=(24, 2)),
            search=replace(baseline().calibration.model, covariance_type="tied"),
            config=GaussianCalibrationConfig(True, n_simulations=1),
        )


def test_joint_significance_accepts_prespecified_family_and_preserves_full_default():
    original = baseline()
    tied = replace(
        original,
        calibration=replace(
            original.calibration,
            model=replace(original.calibration.model, covariance_type="tied"),
        ),
    )
    assert JointStudyConfig.from_dict(tied.to_dict()) == tied
    assert tied.calibration.to_dict()["model"]["covariance_type"] == "tied"
    assert "covariance_type" not in original.to_dict()["calibration"]["model"]
