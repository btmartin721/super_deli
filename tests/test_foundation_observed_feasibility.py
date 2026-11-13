"""Dependency-free fixed design, scheduling limits and optional-import boundaries."""

import pytest

from super_deli.benchmarking.observed_feasibility_config import (
    THRESHOLDS,
    TREE_PARAMETERS,
    ObservedFeasibilityStudy,
    counts,
    model_for,
    roles,
    tasks,
)


@pytest.mark.parametrize(
    "values",
    [
        {"max_projected_seconds": 1801},
        {"preflight_max_seconds": 601},
        {"max_workers": 3},
        {"artifact_cap_bytes": 3 * 1024**3 + 1},
        {"schema_version": True},
        {"fixture": 1},
        {"max_workers": False},
        {"min_free_bytes": 1},
    ],
)
def test_strict_limits(values):
    with pytest.raises(ValueError):
        ObservedFeasibilityStudy(**values)


def test_config_roundtrip_and_unknown_fields():
    c = ObservedFeasibilityStudy()
    assert ObservedFeasibilityStudy.from_dict(c.to_dict()) == c
    with pytest.raises(ValueError):
        ObservedFeasibilityStudy.from_dict({"detector": "different"})
    assert TREE_PARAMETERS["n_estimators"] == 100 and TREE_PARAMETERS["max_depth"] == 3
    assert TREE_PARAMETERS["min_samples_leaf"] == 10 and TREE_PARAMETERS["n_iter_no_change"] is None
    assert model_for("diagnostic", 0).to_dict() == TREE_PARAMETERS
    with pytest.raises(ValueError):
        model_for("neural", 0)
    assert THRESHOLDS == tuple(i / 100 for i in range(101))


def test_exact_scientific_budget():
    config = ObservedFeasibilityStudy()
    a, b = counts(config, "preflight"), counts(config, "main")
    assert (a["observations"], a["workflows"], a["detector_fits"], a["trees"]) == (8, 12, 10, 1000)
    assert (b["observations"], b["workflows"], b["detector_fits"], b["trees"]) == (80, 92, 84, 8400)
    assert a["cohorts"] + b["cohorts"] == 902
    assert a["rows"] + b["rows"] == 720896
    assert a["neural_fits"] == b["neural_fits"] == 0
    assert a["scheduled_frozen_replays"] == 2 and b["scheduled_frozen_replays"] == 8


def test_role_and_phase_separation():
    config = ObservedFeasibilityStudy()
    seen = set()
    for phase in ("preflight", "main"):
        for t in tasks(config, phase):
            for i, r in enumerate(roles(t)):
                key = (*t["seed_path"], i)
                assert key not in seen
                seen.add(key)
                assert key[:2] == (20261112, 750)
                assert r["name"] not in ("size", "latent", "truth")
                if r["stage"] != "test":
                    assert r["group"] in ("known", "near", "far")
    assert len(seen) == 902
    with pytest.raises(ValueError):
        tasks(config, "unrecognized")


def test_invented_budget_is_small_and_separate():
    c = ObservedFeasibilityStudy(fixture=True)
    assert counts(c, "main")["observations"] == 1
    assert counts(c, "preflight")["detector_fits"] == 2
    assert counts(c, "main")["workflows"] == 4


def test_optional_library_blocker():
    import sys

    from super_deli.benchmarking.observed_feasibility_runtime import scientific_runtime

    if any(k.split(".", 1)[0] in ("torch", "snpio") for k in sys.modules):
        pytest.skip("Engineering process already owns an optional library")
    with scientific_runtime():
        with pytest.raises(ImportError):
            __import__("torch")
        with pytest.raises(ImportError):
            __import__("snpio")
