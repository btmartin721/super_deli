"""Frozen budgets and narrow study bounds remain dependency-free."""

from dataclasses import replace

import pytest

from super_deli.benchmarking.neural_calibration_config import NeuralCalibrationStudy, counts, tasks


def test_fixed_full_settings_budget():
    c = NeuralCalibrationStudy()
    assert counts(c, "main")["observations"] == 40
    assert counts(c, "main")["workflows"] == 8008
    assert counts(c, "main")["neural_fits"] == 8008
    assert counts(c, "preflight")["workflows"] == 800
    assert counts(c, "preflight")["neural_fits"] == 800
    assert all(t["n"] == 64 and t["n_simulations"] == 99 for t in tasks(c, "main"))


@pytest.mark.parametrize(
    "kw",
    [
        {"max_workers": 3},
        {"artifact_cap_bytes": 8 * 1024**3},
        {"max_projected_seconds": 14401},
        {"min_free_bytes": 1},
    ],
)
def test_resource_bounds(kw):
    with pytest.raises(ValueError):
        replace(NeuralCalibrationStudy(), **kw)


def test_disjoint_fresh_random_paths():
    c = NeuralCalibrationStudy()
    main = tasks(c, "main")
    pre = tasks(c, "preflight")
    for field in (
        "data_seed",
        "noise_seed",
        "permutation_seed",
        "ecology_order_seed",
        "null_seed_path",
    ):
        assert not {tuple(t[field]) for t in main} & {tuple(t[field]) for t in pre}
