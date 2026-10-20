"""Prevent ambiguous or fabricated uncertainty reporting in later model adapters."""

import math

import pytest

from super_deli.core.results import Assignment, DelimitationResult, Evidence


def test_result_can_explicitly_leave_units_and_support_unresolved():
    result = DelimitationResult(
        "example",
        "future_estimator",
        "specimen",
        (Assignment("s1", None, "Insufficient evidence"),),
        (Evidence("membership_probability", "s1", "not_computed", reason="No estimator fitted"),),
        "a" * 64,
    )
    assert result.to_dict()["assignments"][0]["label"] is None
    assert result.evidence[0].value is None


@pytest.mark.parametrize(
    "kind,extras",
    [
        ("bootstrap_frequency", {"value": 0.8, "replicates": 100}),
        ("membership_probability", {"value": 0.8}),
        ("confidence_interval", {"interval": (1.1, 2.3), "confidence_level": 0.95}),
        ("p_value", {"value": 0.02, "null_hypothesis": "No association", "multiplicity": "none"}),
    ],
)
def test_distinct_evidence_types_are_representable(kind, extras):
    estimate = Evidence(kind, "partition", "available", method="illustrative method", **extras)
    assert estimate.kind == kind


@pytest.mark.parametrize(
    "kwargs",
    [
        {"value": 1.1},
        {"value": -0.1},
        {"value": math.nan},
        {"value": True},
        {"value": math.inf},
        {"value": 0.5, "method": None},
        {"value": 0.5, "status": "not_computed", "reason": "Skipped"},
        {"status": "not_computed"},
        {"value": 0.5, "confidence_level": 0.95},
        {"value": 0.5, "status": "unknown"},
    ],
)
def test_malformed_or_uncomputed_estimates_cannot_masquerade_as_support(kwargs):
    values = dict(kind="membership_probability", target="s1", status="available", method="test")
    values.update(kwargs)
    with pytest.raises(ValueError):
        Evidence(**values)


def test_significance_requires_null_and_multiplicity_metadata():
    with pytest.raises(ValueError, match="null_hypothesis"):
        Evidence("p_value", "partition", "available", method="test", value=0.04)
    with pytest.raises(ValueError, match="multiplicity"):
        Evidence(
            "p_value",
            "partition",
            "available",
            method="test",
            value=0.04,
            null_hypothesis="No association",
        )


def test_bootstrap_requires_replicates():
    with pytest.raises(ValueError, match="replicates"):
        Evidence("bootstrap_frequency", "partition", "available", method="test", value=0.9)


@pytest.mark.parametrize(
    "interval,confidence", [((2, 1), 0.95), ((0, math.nan), 0.95), ((0, 1), 1), ((0, 1), None)]
)
def test_intervals_require_finite_ordered_endpoints_and_coverage(interval, confidence):
    with pytest.raises(ValueError):
        Evidence(
            "confidence_interval",
            "parameter",
            "available",
            method="test",
            interval=interval,
            confidence_level=confidence,
        )


def test_repeated_observations_cannot_create_duplicate_unit_assignments():
    with pytest.raises(ValueError, match="same biological unit"):
        DelimitationResult(
            "example",
            "future_estimator",
            "specimen",
            (Assignment("s1", "a"), Assignment("s1", "b")),
            (Evidence("membership_probability", "s1", "unavailable", reason="Not supported"),),
            "a" * 64,
        )


def test_unresolved_assignments_require_explanation():
    with pytest.raises(ValueError, match="unresolved"):
        Assignment("s1", None)
