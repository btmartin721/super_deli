"""Dependency-free source-binding and CLI declaration contracts."""

from dataclasses import replace

import pytest

from super_deli.benchmarking.genomic_diagnosis_config import GenomicDiagnosisConfig


def config():
    return GenomicDiagnosisConfig("0" * 64, "1" * 64, "2" * 40)


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"source_manifest_sha256": "x" * 64},
        {"source_completion_sha256": "A" * 64},
        {"source_commit": "abc"},
        {"source_role": "guess"},
        {"source_version": "next"},
        {"expected_observations": 47},
        {"expected_subsample_models": -1},
        {"max_output_bytes": 0},
        {"max_file_bytes": True},
    ],
)
def test_invalid(changes):
    with pytest.raises(ValueError):
        replace(config(), **changes)


def test_roundtrip():
    value = config()
    assert GenomicDiagnosisConfig.from_dict(value.to_dict()) == value
    with pytest.raises(ValueError):
        GenomicDiagnosisConfig.from_dict({**value.to_dict(), "tune": True})


def test_lazy_help(capsys):
    from super_deli.cli import main

    with pytest.raises(SystemExit) as error:
        main(["diagnose-genomic-neural", "--help"])
    assert error.value.code == 0 and "--diagnosis-config" in capsys.readouterr().out
