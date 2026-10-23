"""Dependency-free reconstruction of published four-arm recovery and support tables."""

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/grouped-pca/results"


def read(name):
    return json.loads((RESULTS / name).read_text())


def rows(name):
    with (RESULTS / name).open() as handle:
        return list(csv.DictReader(handle))


def test_manifest_frozen_settings_and_explicit_new_option():
    manifest = read("manifest.json")
    assert manifest["local_artifact_id"] == "grouped-pca-20261023"
    assert set(manifest["published_artifacts"]) == {p.name for p in RESULTS.iterdir()} - {
        "manifest.json"
    }
    for name, digest in manifest["published_artifacts"].items():
        assert hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["figures"].items():
        assert (
            hashlib.sha256((ROOT / "docs/figures/grouped-pca" / name).read_bytes()).hexdigest()
            == digest
        )
    c = read("configuration.json")
    assert (c["random_seed"], c["n_datasets"], c["n_stability"], c["n_bootstrap"]) == (
        20261106,
        25,
        5,
        25,
    )
    assert all(v.get("pca_fit", "source_rows") == "source_rows" for v in c["model"]["views"])
    for arm, model in read("grid.json").items():
        assert (
            model["n_pcs"] == int(arm[-1])
            and model["n_bootstrap"] == 0
            and model["covariance_type"] == "tied"
        )
        assert model["views"][0].get("pca_fit", "source_rows") == (
            "target_means" if arm.startswith("target") else "source_rows"
        )
    assert not any("/Users/" in p.read_text() for p in RESULTS.iterdir())


def test_recovery_counts_intervals_ari_and_paired_differences():
    points = rows("points.csv")
    summary = read("summary.json")
    assert len(points) == summary["successful_points"] == summary["planned_points"] == 700
    assert summary["failed_points"] == summary["significance_fits"] == 0
    assert summary["reproduced_archived"] == 25 and len(summary["summaries"]) == 28
    assert len({(r["cohort"], r["repetition"], r["arm"]) for r in points}) == 700
    for group in summary["summaries"]:
        selected = [r for r in points if (r["cohort"], r["arm"]) == (group["cohort"], group["arm"])]
        n = len(selected)
        assert n == 25 and group["available"] == n
        counts = {str(k): sum(r["selected_k"] == str(k) for r in selected) for k in range(1, 5)}
        assert counts == group["selected_k_counts"]
        for name, events in (("four", counts["4"]), ("non_single", n - counts["1"])):
            rate = group["rates"][name]
            p, z = events / n, 1.959963984540054
            center = (p + z * z / (2 * n)) / (1 + z * z / n)
            radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            assert rate["events"] == events and rate["rate"] == p
            assert rate["wilson_95"] == pytest.approx(
                [max(0, center - radius), min(1, center + radius)]
            )
        for key in ("ari", "joint_a_r2", "joint_b_r2", "k4_minus_k2_bic"):
            values = [float(r[key]) for r in selected if r[key]]
            if values:
                assert group[key]["median"] == pytest.approx(statistics.median(values))
                assert group[key]["mean"] == pytest.approx(statistics.mean(values))
                assert group[key]["mean_mcse"] == pytest.approx(
                    statistics.stdev(values) / math.sqrt(len(values))
                )
            else:
                assert group[key] is None
    for group in summary["paired"]:
        a = {
            r["repetition"]: r
            for r in points
            if r["cohort"] == group["cohort"] and r["arm"] == group["left"]
        }
        b = {
            r["repetition"]: r
            for r in points
            if r["cohort"] == group["cohort"] and r["arm"] == group["right"]
        }
        differences = [float(b[i]["ari"]) - float(a[i]["ari"]) for i in a if a[i]["ari"]]
        assert group["available_pairs"] == 25
        if differences:
            assert group["mean_ari_change"] == pytest.approx(statistics.mean(differences))
        else:
            assert group["mean_ari_change"] is None
        assert group["gained_four"] == sum(
            a[i]["selected_k"] != "4" and b[i]["selected_k"] == "4" for i in a
        )
        assert group["lost_four"] == sum(
            a[i]["selected_k"] == "4" and b[i]["selected_k"] != "4" for i in a
        )


def test_tied_bic_counts_and_no_group_truth_is_withheld():
    candidates = rows("candidates.csv")
    assert len(candidates) == 2800
    for row in candidates:
        n, d, k = (int(row[key]) for key in ("n", "d", "k"))
        parameters = k - 1 + k * d + d * (d + 1) // 2
        assert int(row["parameters"]) == parameters
        assert float(row["penalty"]) == pytest.approx(parameters * math.log(n))
        assert float(row["bic"]) == pytest.approx(
            float(row["penalty"]) - 2 * float(row["log_likelihood_from_bic"])
        )
    for row in rows("points.csv"):
        if row["cohort"] in {"independent_gaussian", "correlated_gaussian", "cline"}:
            assert row["ari"] == row["joint_a_r2"] == row["joint_b_r2"] == ""


def test_stability_denominators_and_postfit_truth_summaries():
    summary = read("summary.json")
    stability = rows("stability.csv")
    assert len(stability) == 40 and summary["planned_bootstrap"] == 1000
    assert sum(int(r["attempted"]) for r in stability) == summary["attempted_bootstrap"] == 1000
    assert sum(int(r["successful"]) for r in stability) == summary["successful_bootstrap"] == 1000
    assert summary["failed_bootstrap"] == summary["unattempted_bootstrap"] == 0
    replicas = rows("stability_replicates.csv")
    assert len(replicas) == 1000
    assert all(r["status"] == "complete" and int(r["n_oob"]) > 1 for r in replicas)
    quality = rows("stability_quality.csv")
    assert len(quality) == 40
    for row in quality:
        selected = [
            r for r in replicas if all(r[k] == row[k] for k in ("cohort", "repetition", "arm"))
        ]
        assert len(selected) == int(row["requested"]) == int(row["available"]) == 25
        for key in ("ari_all", "ari_oob"):
            values = [float(r[key]) for r in selected]
            assert float(row[f"mean_{key}"]) == pytest.approx(statistics.mean(values))
            assert float(row[f"median_{key}"]) == pytest.approx(statistics.median(values))
        for pair in ("within", "between"):
            assert 0 <= float(row[f"mean_oob_coassignment_{pair}"]) <= 1
            assert (
                0 < int(row[f"oob_pairs_available_{pair}"]) <= int(row[f"oob_pairs_total_{pair}"])
            )
