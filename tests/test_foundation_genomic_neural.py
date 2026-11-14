"""Dependency-free genomic neural declarations and lazy command contracts."""

import pytest

from super_deli.models.genomic_neural_config import GenomicComparisonConfig, GenomicNeuralConfig


@pytest.mark.parametrize(
    "changes",
    [
        {"ploidy": 1},
        {"missing_policy": "mean"},
        {"assume_independent_specimens": False},
        {"n_pcs": 3},
        {"schema_version": 2},
        {"min_cluster_size": 2},
        {"random_seed": True},
        {"encoding_seed": -1},
        {"max_loci": 4097},
        {"n_subsamples": -1},
        {"max_working_bytes": 0},
    ],
)
def test_reject_invalid(changes):
    values = dict(ploidy=2, missing_policy="error", assume_independent_specimens=True)
    values.update(changes)
    with pytest.raises(ValueError):
        GenomicNeuralConfig(**values)


def test_roundtrip_and_legacy_defaults():
    from super_deli.models.config import BaselineConfig

    old = BaselineConfig(2, "error", True)
    new = GenomicNeuralConfig(2, "error", True)
    assert old.n_bootstrap == 100 and new.baseline_config().n_bootstrap == 0
    assert GenomicNeuralConfig.from_dict(new.to_dict()) == new
    config = GenomicComparisonConfig(new)
    assert GenomicComparisonConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError):
        GenomicComparisonConfig(new, (0, 0))
    with pytest.raises(ValueError):
        GenomicNeuralConfig.from_dict({**new.to_dict(), "unknown": 1})


def test_lazy_cli(capsys):
    from super_deli.cli import main

    with pytest.raises(SystemExit) as error:
        main(["analyze-genomic-models", "--help"])
    assert error.value.code == 0
    assert "--model-config" in capsys.readouterr().out
