"""Abstention is explicit, separately versioned and dependency-free to configure."""

import json

import pytest

from super_deli.benchmarking.abstention_config import AbstentionStudy, counts, tasks
from super_deli.models.reference_abstention_config import ReferenceAbstentionConfig


def test_explicit_contract_and_legacy_study_budgets():
    config = ReferenceAbstentionConfig(True)
    assert ReferenceAbstentionConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    assert config.policy == "distance"
    study = AbstentionStudy()
    assert len(tasks(study, "main")) == 80
    assert counts(study, "main")["classifier_trainings"] == 320
    assert counts(study, "main")["workflows"] == 328
    assert counts(study, "main")["verified_updates"] == 120000
    assert counts(study, "preflight")["workflows"] == 32
    assert counts(study, "preflight")["verified_updates"] == 12000
    assert counts(study, "main")["policy_evaluations"] == 4080
    assert counts(study, "preflight")["policy_evaluations"] == 408


@pytest.mark.parametrize(
    "options",
    [
        {},
        dict(assume_independent_calibration=False),
        dict(assume_independent_calibration=1),
        dict(assume_independent_calibration=True, policy="automatic"),
        dict(assume_independent_calibration=True, exclusion_level=0.1),
        dict(assume_independent_calibration=True, covariance_shrinkage=0),
        dict(assume_independent_calibration=True, schema_version=True),
        dict(assume_independent_calibration=True, unknown=True),
    ],
)
def test_invalid_or_implicit_policies(options):
    with pytest.raises(ValueError):
        ReferenceAbstentionConfig.from_dict(options)


@pytest.mark.parametrize(
    "command",
    [
        "calibrate-reference-abstention",
        "predict-reference-abstention",
        "benchmark-reference-abstention",
    ],
)
def test_cli_help_without_optional_dependencies(command, capsys):
    from super_deli.cli import main

    with pytest.raises(SystemExit) as exit:
        main([command, "--help"])
    assert exit.value.code == 0
    assert "--output-dir" in capsys.readouterr().out
