"""Dependency-free strict known-only contracts and exact study counts."""

import pytest

from super_deli.benchmarking.known_reference_config import KnownReferenceStudy, counts, tasks
from super_deli.models.reference_density_config import ReferenceDensityConfig


@pytest.mark.parametrize(
    "options",
    [
        {"assume_independent_calibration": False},
        {"assume_independent_calibration": 1},
        {"assume_independent_calibration": True, "schema_version": True},
        {"assume_independent_calibration": True, "exclusion_level": 0.1},
        {"assume_independent_calibration": True, "covariance_shrinkage": 0},
    ],
)
def test_density_contract_rejects_unapproved_settings(options):
    with pytest.raises(ValueError):
        ReferenceDensityConfig(**options)


def test_config_roundtrip_and_unknown_keys():
    c = ReferenceDensityConfig(True)
    assert ReferenceDensityConfig.from_dict(c.to_dict()) == c
    with pytest.raises(ValueError):
        ReferenceDensityConfig.from_dict({**c.to_dict(), "unknown_examples": []})
    with pytest.raises(ValueError):
        KnownReferenceStudy.from_dict({"best_seed": 101})
    with pytest.raises(ValueError):
        KnownReferenceStudy(artifact_cap_bytes=7 * 1024**3)
    with pytest.raises(ValueError):
        KnownReferenceStudy(max_workers=3)


def test_exact_scientific_schedule_and_repeated_measurements():
    c = KnownReferenceStudy()
    a = counts(c, "preflight")
    b = counts(c, "main")
    assert (a["observations"], a["work_units"], a["workflows"], a["verified_updates"]) == (
        8,
        10,
        68,
        17000,
    )
    assert (b["observations"], b["work_units"], b["workflows"], b["verified_updates"]) == (
        80,
        84,
        648,
        162000,
    )
    assert a["rows"] + b["rows"] == 523776 and a["cohorts"] + b["cohorts"] == 550
    assert len({tuple(t["seed_path"]) for t in tasks(c, "main")}) == 80
    with pytest.raises(ValueError):
        tasks(c, "other")
