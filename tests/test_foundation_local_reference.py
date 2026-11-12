"""Dependency-free versioned contracts; no legacy policy reinterpretation."""

import pytest

from super_deli.models.local_reference_config import LocalReferenceSupportConfig
from super_deli.models.reference_abstention_config import ReferenceAbstentionConfig


def test_separate_serialization_and_legacy_defaults():
    old = ReferenceAbstentionConfig(True).to_dict()
    cfg = LocalReferenceSupportConfig(True)
    assert LocalReferenceSupportConfig.from_dict(cfg.to_dict()) == cfg
    assert old == ReferenceAbstentionConfig(True).to_dict()
    assert old["policy"] == "distance" and cfg.neighbors == 5


@pytest.mark.parametrize(
    "changes",
    [
        dict(schema_version=True),
        dict(schema_version=2),
        dict(neighbors=True),
        dict(neighbors=4),
        dict(exclusion_level=0.1),
        dict(covariance_shrinkage=0),
        dict(assume_independent_calibration=False),
        dict(assume_independent_calibration=1),
    ],
)
def test_invalid_frozen_settings(changes):
    with pytest.raises(ValueError):
        LocalReferenceSupportConfig(**(dict(assume_independent_calibration=True) | changes))


@pytest.mark.parametrize(
    "value", [{}, {"assume_independent_calibration": True, "policy": "distance"}, None]
)
def test_unknown_or_missing_fields(value):
    with pytest.raises(ValueError):
        LocalReferenceSupportConfig.from_dict(value)


def test_study_budget_and_legacy_ceiling():
    from super_deli.benchmarking.local_reference_config import LocalReferenceStudy, counts, tasks
    from super_deli.benchmarking.residual_config import ResidualStudy

    cfg = LocalReferenceStudy()
    assert cfg.artifact_cap_bytes == 3 * 1024**3
    assert counts(cfg, "preflight")["workflows"] == 64
    assert counts(cfg, "main")["workflows"] == 616
    assert counts(cfg, "preflight")["policy_evaluations"] == 1088
    assert counts(cfg, "main")["policy_evaluations"] == 10336
    assert counts(cfg, "main")["oracle_evaluations"] == 12274
    assert all(
        t["seed_path"][:2] == [20261112, 740] and t["key"].startswith("local_")
        for t in tasks(cfg, "main")
    )
    with pytest.raises(ValueError):
        LocalReferenceStudy(artifact_cap_bytes=4 * 1024**3)
    with pytest.raises(ValueError):
        LocalReferenceStudy(max_workers=3)
    with pytest.raises(ValueError):
        ResidualStudy(artifact_cap_bytes=3 * 1024**3)


def test_local_commands_stay_dependency_free():
    import subprocess
    import sys

    script = (
        "import sys;from super_deli.cli import main\n"
        'try: main(["calibrate-local-reference-support","--help"])\n'
        "except SystemExit as exc: assert exc.code == 0\n"
        'assert "numpy" not in sys.modules'
    )
    # argparse intentionally exits after help; foundation environments have no numerical extras.
    p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert p.returncode == 0 and "--support-config" in p.stdout
