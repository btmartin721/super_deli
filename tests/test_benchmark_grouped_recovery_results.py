"""Dependency-free reconstruction of published diagnostic counts and BIC terms."""

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/grouped-recovery/results"


def read(name):
    return json.loads((RESULTS / name).read_text())


def rows(name):
    with (RESULTS / name).open() as handle:
        return list(csv.DictReader(handle))


def test_manifest_and_frozen_configuration():
    manifest = read("manifest.json")
    assert manifest["local_artifact_id"] == "grouped-recovery-20261023"
    assert set(manifest["published_artifacts"]) == {p.name for p in RESULTS.iterdir()} - {
        "manifest.json"
    }
    for name, expected in manifest["published_artifacts"].items():
        assert hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == expected
    for name, expected in manifest["figures"].items():
        assert (
            hashlib.sha256((ROOT / "docs/figures/grouped-recovery" / name).read_bytes()).hexdigest()
            == expected
        )
    assert read("configuration.json")["random_seed"] == 20261104
    assert read("runtime_versions.json")["super-deli"] == "0.26.0.dev0"
    assert all(
        m["covariance_type"] == "tied" and m["n_bootstrap"] == 0 for m in read("grid.json").values()
    )
    assert not any("/Users/" in p.read_text() for p in RESULTS.iterdir())


def test_point_counts_rates_intervals_and_metrics():
    points, summary = rows("points.csv"), read("summary.json")
    assert len(points) == summary["planned_points"] == 750
    assert summary["successful_points"] == 700 and summary["failed_points"] == 50
    assert summary["replayed_archived"] == 25 and summary["bootstrap_or_calibration_fits"] == 0
    assert len({(r["cohort"], r["repetition"], r["variant"]) for r in points}) == 750
    for result in summary["summaries"]:
        selected = [
            r
            for r in points
            if (r["cohort"], r["variant"]) == (result["cohort"], result["variant"])
        ]
        available = [r for r in selected if r["status"] == "complete"]
        assert len(selected) == result["requested"] == 25
        assert len(available) == result["available"] and 25 - len(available) == result["failed"]
        assert result["selected_k_counts"] == {
            str(k): sum(r["selected_k"] == str(k) for r in available) for k in range(1, 5)
        }
        if not available:
            assert result["cohort"] == "fresh_256" and result["variant"].startswith("views_4_")
            assert (
                result["selected_four_rate"]
                is result["selected_four_wilson_95"]
                is result["ari"]
                is None
            )
            continue
        n, events = len(available), sum(r["selected_k"] == "4" for r in available)
        probability, z = events / n, 1.959963984540054
        center = (probability + z * z / (2 * n)) / (1 + z * z / n)
        radius = (
            z
            * math.sqrt(probability * (1 - probability) / n + z * z / (4 * n * n))
            / (1 + z * z / n)
        )
        assert result["selected_four_rate"] == probability
        assert result["selected_four_wilson_95"] == pytest.approx(
            [max(0, center - radius), min(1, center + radius)]
        )
        for key in (
            "ari",
            "fused_b_r2",
            "joint_b_r2",
            "morphological_profile_a_r2",
            "ecological_profile_b_r2",
        ):
            values = [float(r[key]) for r in available]
            assert result[key]["median"] == pytest.approx(statistics.median(values))
            assert result[key]["mean"] == pytest.approx(statistics.mean(values))


def test_shared_family_bic_decompositions():
    candidates = rows("candidates.csv")
    assert len(candidates) == 2800
    for row in candidates:
        n, d, k = (int(row[key]) for key in ("n", "d", "k"))
        parameters = k - 1 + k * d + d * (d + 1) // 2
        assert int(row["parameters"]) == parameters
        assert float(row["penalty"]) == pytest.approx(parameters * math.log(n))
        if row["bic"]:
            assert float(row["bic"]) == pytest.approx(
                -2 * float(row["log_likelihood_from_bic"]) + float(row["penalty"])
            )
            assert float(row["bic_minus_selected"]) == pytest.approx(
                float(row["penalty_minus_selected"]) - float(row["twice_log_likelihood_gain"]),
                abs=1e-9,
            )
    baseline = [r for r in rows("points.csv") if r["variant"] == "baseline"]
    assert len(baseline) == 75 and all(r["k4_status"] == "eligible" for r in baseline)


def test_truth_diagnostics_and_arithmetic_failures_stay_separate():
    truth = rows("truth_initialized.csv")
    assert len(truth) == 75 and all(r["status"] == "complete" for r in truth)
    baseline = {
        (r["cohort"], r["repetition"]): r for r in rows("points.csv") if r["variant"] == "baseline"
    }
    assert not any(
        r["eligible"] == "True"
        and float(r["bic_minus_baseline_selected"]) < 0
        and baseline[r["cohort"], r["repetition"]]["selected_k"] != "4"
        for r in truth
    )
    checks = read("arithmetic-audit.json")["records"]
    assert len(checks) == 50 and all(
        r["finite"] and r["rank"] == 4 and r["matmul_output_finite"] for r in checks
    )
    assert sum(r["strict_matmul"] == "failed" for r in checks) == 25
    assert max(r["maximum_difference_from_explicit_sum"] for r in checks) < 1e-15
