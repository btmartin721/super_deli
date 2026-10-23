"""Dependency-free equations and provenance for the frozen boundary diagnostic."""

import csv
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

import pytest

from super_deli.benchmarking.config import derived_seed

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/grouped-boundary-diagnostics/results"


def read(name):
    return json.loads((RESULTS / name).read_text())


def rows(name):
    opener = gzip.open if name.endswith(".gz") else open
    with opener(RESULTS / name, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def pipelines():
    return [r for p in sorted(RESULTS.glob("pipelines-*.csv.gz")) for r in rows(p.name)]


def wilson(x, n):
    if n == 0:
        return None
    z, p = 1.959963984540054, x / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0, mid - half), min(1, mid + half)]


def test_hashes_privacy_preservation_and_unchanged_production():
    m = read("manifest.json")
    assert m["local_artifact_id"] == "grouped-boundary-diagnostics-20261023"
    for name, digest in m["published_artifacts"].items():
        path = RESULTS / name
        assert Path(name).name == name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        text = gzip.open(path, "rt").read() if name.endswith(".gz") else path.read_text()
        assert "/Users/" not in text and "/private/tmp" not in text
    for name, digest in m["figure_files"].items():
        assert (
            hashlib.sha256(
                (ROOT / "docs/figures/grouped-boundary-diagnostics" / name).read_bytes()
            ).hexdigest()
            == digest
        )
    for key, relative in (
        ("report_script_sha256", "scripts/report_grouped_boundary_diagnostics.py"),
        ("independent_checker_sha256", "examples/grouped-boundary-diagnostics/check_tails.py"),
        ("independent_candidate_equations_sha256", "examples/grouped-likelihood/check_tails.py"),
    ):
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == m[key]
    changes = read("source-changes.json")
    assert (changes["prior_modules"], changes["unchanged"]) == (134, 132)
    assert set(changes["changed"]) == {"__init__.py", "cli.py"}
    assert len(changes["new_modules"]) == 4
    frozen, base = read("runtime_sources.json"), read("base_sources.json")
    assert len(frozen) == 138 and len(base) == 134
    assert all(frozen[n] == digest for n, digest in base.items() if n not in changes["changed"])
    assert read("runtime_versions.json")["libraries"]["super-deli"]["version"] == "0.32.0.dev0"
    preserved = read("preservation-summary.json")
    assert preserved["before_sha256"] == preserved["after_sha256"]
    assert preserved["prior_study_files"] == 188822 and preserved["checkouts"] == 6
    assert preserved["original_tps_files"] == 5 and preserved["aic_metadata_files"] == 1078


def test_frozen_design_seeds_and_unique_work_accounting(pipelines):
    design = read("prespecified-design.json")
    assert (design["master_seed"], design["model_seed"]) == (20261116, 71323)
    assert design["n_datasets_per_case"] == 25 and design["n_units"] == 64
    assert design["draws_per_arm"] == 99 and design["new_unique_pipeline_fits"] == 24875
    assert design["alpha"] == 0.05 and len(design["seeds"]) == 125
    assert len({r["data_seed"] for r in design["seeds"]}) == 125
    for r in design["seeds"]:
        for field, label in (("data_seed", "observed_data"), ("null_stream_seed", "null_stream")):
            assert r[field] == derived_seed(20261116, r["case"], r["repetition"], label)
    assert read("seed-domain-check.json") == {
        "fresh_observations": 125,
        "prior_observations": 250,
        "disjoint": True,
        "distinct_fresh_seeds": 125,
    }
    summary = read("summary.json")
    assert summary["new_unique_planned_fits"] == 24875
    assert len(pipelines) == summary["unique_pipeline_records_audited"]
    assert len(
        {(p["cohort"], p["case"], p["repetition"], p["arm"], p["replicate"]) for p in pipelines}
    ) == len(pipelines)
    cases = summary["cases"]
    assert len(cases) == 5
    assert sum(s["planned_fits"] for s in cases) == 24875
    assert sum(s["successful_fits"] for s in cases) == summary["new_unique_successful_fits"]
    for s in cases:
        assert s["successful_fits"] + s["failed_fits"] + s["unattempted_fits"] == s["planned_fits"]
        assert s["attempted_fits"] == s["successful_fits"] + s["failed_fits"]
        assert s["replacement_draws"] == s["replacement_observations"] == 0
    execution = read("study-execution.json")
    assert execution["workers"] == 4 and execution["threads_per_job"] == 1
    assert execution["status"] == "complete"
    assert execution["planned_unique_pipeline_fits"] == 24875


def test_every_tail_and_interval_against_portable_pipeline_values(pipelines):
    by_test = {}
    observed = {}
    for p in pipelines:
        key = (p["cohort"], p["case"], p["repetition"], p["arm"])
        if int(p["replicate"]) == -1:
            observed[key] = p
        else:
            by_test.setdefault(key, []).append(p)
    tail_rows = rows("tails.csv")
    assert len(tail_rows) == 750
    for r in tail_rows:
        key = (r["cohort"], r["case"], r["repetition"], r["arm"])
        point = observed.get((*key[:3], "fitted"))
        if not r["p_value"]:
            assert r["reject"] == ""
            continue
        assert point is not None and point["status"] == "complete"
        field = "statistic" if r["statistic_kind"] == "likelihood" else "bic_statistic"
        draws = by_test[key]
        assert len(draws) == 99 and all(d["status"] == "complete" for d in draws)
        x = sum(float(d[field]) >= float(point[field]) for d in draws)
        assert int(r["exceedances"]) == x and float(r["statistic"]) == float(point[field])
        assert float(r["p_value"]) == (1 + x) / 100
        assert (r["reject"] == "True") == ((1 + x) / 100 <= 0.05)
        low, high = json.loads(r["conditional_tail_interval_95"])

        def probability(p, lo, hi):
            return math.fsum(
                math.comb(99, k) * p**k * (1 - p) ** (99 - k) for k in range(lo, hi + 1)
            )

        assert low == 0 if x == 0 else probability(low, x, 99) == pytest.approx(0.025, abs=1e-10)
        assert high == 1 if x == 99 else probability(high, 0, x) == pytest.approx(0.025, abs=1e-10)
    independent = read("independent-tails.json")
    assert independent["observations"] == 250 and independent["tails_checked"] == 750
    assert independent["verified_unique_pipeline_statistics"] == sum(
        p["status"] == "complete" for p in pipelines
    )


def test_rates_and_paired_discordance_use_observations():
    data = rows("tails.csv")
    summary = read("summary.json")
    assert len(summary["summaries"]) == 30
    for s in summary["summaries"]:
        group = [
            r
            for r in data
            if all(r[k] == s[k] for k in ("cohort", "case", "arm", "statistic_kind"))
        ]
        available = [r for r in group if r["p_value"]]
        x = sum(r["reject"] == "True" for r in available)
        assert len(group) == s["requested"] == 25
        assert len(available) == s["available"] and x == s["rejected"]
        assert s["wilson_95"] == pytest.approx(wilson(x, len(available)), abs=1e-12)
        assert s["withheld_bounds"] == [x / 25, (x + 25 - len(available)) / 25]
        assert all(v is None or "mean_mcse" not in v for v in s["null_pipeline_metrics"].values())
    for s in summary["paired_changes"]:
        left = {
            r["repetition"]: r
            for r in data
            if r["cohort"] == "fresh"
            and r["case"] == s["case"]
            and r["arm"] == "fitted"
            and r["statistic_kind"] == s["statistic_kind"]
        }
        right = {
            r["repetition"]: r
            for r in data
            if r["cohort"] == "fresh"
            and r["case"] == s["case"]
            and r["arm"] == "known"
            and r["statistic_kind"] == s["statistic_kind"]
        }
        ids = [i for i in left if left[i]["p_value"] and right[i]["p_value"]]
        assert len(ids) == s["available_pairs"]
        assert (
            sum(left[i]["reject"] == "True" and right[i]["reject"] != "True" for i in ids)
            == s["fitted_only_reject"]
        )
        assert (
            sum(right[i]["reject"] == "True" and left[i]["reject"] != "True" for i in ids)
            == s["known_only_reject"]
        )


def test_variance_law_and_clustered_uncertainty(pipelines):
    covariance = rows("covariance.csv")
    assert 0 < len(covariance) <= 250
    for r in covariance:
        trace = float(r["true_conditional_trace"])
        assert float(r["true_count_variance_ratio"]) == pytest.approx((trace + 4) / (trace + 1))
        expected_rank = {"zero_conditional_gaussian": 0, "rank_two_gaussian": 2}.get(r["case"], 4)
        assert int(r["true_rank"]) == expected_rank and 0 <= int(r["fitted_rank"]) <= 4
        if r["case"] == "zero_conditional_gaussian":
            assert trace == pytest.approx(0, abs=1e-14)
        assert float(r["conditional_error"]) >= 0
    observations = rows("observations.csv")
    groups = read("observation-summary.json")
    assert len(observations) <= 375 and len(groups) <= 15
    for g in groups:
        selected = [r for r in observations if all(r[k] == g[k] for k in ("cohort", "case", "arm"))]
        assert len(selected) == g["n_observations_with_null_fits"] <= 25
        values = [float(r["mean_null_statistic"]) for r in selected if r["mean_null_statistic"]]
        result = g["metrics"]["mean_null_statistic"]
        assert result["n"] == len(values)
        assert result["mean"] == pytest.approx(statistics.mean(values))
        assert result["mean_mcse"] == pytest.approx(
            statistics.stdev(values) / math.sqrt(len(values))
        )
    for r in pipelines:
        if r["status"] == "complete":
            assert 0 <= float(r["count_association"]) <= 1 + 1e-12
            assert 0 <= float(r["lr_count_association"]) <= 1 + 1e-12
            for key in ("morphology_retained_variance", "ecology_retained_variance"):
                assert 0 <= float(r[key]) <= 1 + 1e-12


def test_feasible_known_parameter_likelihood_flags():
    for r in rows("raw-likelihood.csv"):
        gap = float(r["known_negative_log_likelihood"]) - float(r["fitted_negative_log_likelihood"])
        assert float(r["known_minus_fitted"]) == gap
        assert (r["feasible_reference_better"] == "True") == (gap < -1e-6)
