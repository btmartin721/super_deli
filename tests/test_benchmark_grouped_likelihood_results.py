"""Independent equations and immutable provenance for the paired likelihood study."""

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import pytest

from super_deli.benchmarking.config import derived_seed
from super_deli.models.grouped_likelihood_config import CASES, GroupedLikelihoodStudyConfig

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/grouped-likelihood/results"


def read(name):
    return json.loads((RESULTS / name).read_text())


def rows(name):
    with (RESULTS / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def wilson(x, n):
    if not n:
        return None
    z, p = 1.959963984540054, x / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0, mid - half), min(1, mid + half)]


def test_published_hashes_frozen_runtime_and_format_equivalence():
    manifest = read("manifest.json")
    assert manifest["local_artifact_id"] == "aic-grouped-20261023"
    for name, h in manifest["published_artifacts"].items():
        assert (
            Path(name).name == name
            and hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == h
        )
    for name, h in manifest["figure_files"].items():
        assert (
            hashlib.sha256(
                (ROOT / "docs/figures/grouped-likelihood" / name).read_bytes()
            ).hexdigest()
            == h
        )
    assert (
        hashlib.sha256((ROOT / "scripts/report_grouped_likelihood.py").read_bytes()).hexdigest()
        == manifest["report_script_sha256"]
    )
    assert len(read("runtime_sources.json")) == 134
    assert read("runtime_versions.json")["libraries"]["super-deli"]["version"] == "0.31.0.dev0"
    changes = read("source-changes.json")
    assert changes["prior_modules"] == 129 and changes["unchanged"] == 122
    assert len(changes["changed"]) == 7 and len(changes["new_modules"]) == 5
    assert all(r["ast_equivalent"] for r in read("format-equivalence.json")["format_only_changes"])
    assert read("format-equivalence.json")["frozen_runtime_unchanged"] is True
    for p in RESULTS.iterdir():
        assert "/Users/" not in p.read_text() and "/private/tmp" not in p.read_text()


def test_prespecified_seeds_scope_unique_fit_accounting_and_legacy_replay():
    s = GroupedLikelihoodStudyConfig.from_dict(read("study.json"))
    assert (s.random_seed, s.n_datasets, s.n_units, s.large_n) == (20261114, 25, 64, 256)
    m = s.calibration.model
    assert (
        m.selection_criterion,
        m.covariance_type,
        m.n_pcs,
        m.n_init,
        m.max_k,
        m.n_bootstrap,
    ) == ("aic", "full", 4, 10, 4, 0)
    assert m.random_seed == 71323 and s.calibration.n_simulations == 99
    assert all(v.n_pcs == 2 and v.weight == 1 for v in m.views)
    data = rows("datasets.csv")
    assert len(data) == 250 and len({r["data_seed"] for r in data}) == 250
    development = json.loads(
        (ROOT / "examples/full-grouped-diagnostics/results/datasets.json").read_text()
    )
    assert not ({int(r["data_seed"]) for r in data} & {r["seed"] for r in development})
    for r in data:
        assert r["case"] in CASES
        assert int(r["data_seed"]) == derived_seed(
            s.random_seed, r["case"], int(r["repetition"]), "observed_data"
        )
        assert int(r["null_stream_seed"]) == derived_seed(
            s.random_seed, r["case"], int(r["repetition"]), "null_stream"
        )
    summary = read("summary.json")
    for key in (
        "planned_fits",
        "attempted_fits",
        "successful_fits",
        "failed_fits",
        "unattempted_fits",
    ):
        assert summary[key] == sum(int(r[key]) for r in data)
    assert summary["planned_fits"] == 25000
    assert (
        summary["successful_fits"] + summary["failed_fits"] + summary["unattempted_fits"] == 25000
    )
    replay = read("legacy-replay.json")
    assert len(replay) == 25 and all(r["status"] == "reproduced" for r in replay)
    assert not ({r["data_seed"] for r in replay} & {int(r["data_seed"]) for r in data})
    execution = read("study-execution.json")
    assert execution["workers"] == 4 and execution["threads_per_job"] == 1
    assert execution["replacement_observations"] == execution["replacement_draws"] == 0


def test_both_selections_and_statistics_from_independent_candidate_equations():
    candidates = rows("observed_candidates.csv")
    groups = {}
    for c in candidates:
        n, d, k = int(c["n"]), int(c["d"]), int(c["k"])
        q = k - 1 + k * d + k * d * (d + 1) // 2
        assert q == int(c["parameters"])
        if c["log_likelihood"]:
            ll = float(c["log_likelihood"])
            assert float(c["bic"]) == pytest.approx(-2 * ll + q * math.log(n), abs=1e-9)
            assert float(c["aic"]) == pytest.approx(-2 * ll + 2 * q, abs=1e-9)
            assert (int(c["minimum_size"]) >= 5) == (c["status"] == "eligible")
        groups.setdefault((c["case"], c["repetition"]), []).append(c)
    for r in rows("paired.csv"):
        if not r.get("selected_k"):
            continue
        eligible = [c for c in groups[r["case"], r["repetition"]] if c["status"] == "eligible"]
        one = next(c for c in eligible if int(c["k"]) == 1)
        criterion = r["criterion"]
        selected = min(eligible, key=lambda c: (float(c[criterion]), int(c["k"])))
        assert int(r["selected_k"]) == int(selected["k"])
        t = (
            float(one["bic"]) - min(float(c["bic"]) for c in eligible)
            if criterion == "bic"
            else 2
            * (max(float(c["log_likelihood"]) for c in eligible) - float(one["log_likelihood"]))
        )
        assert float(r["statistic"]) == pytest.approx(t, abs=1e-9)


def test_plus_one_tail_and_independent_binomial_interval_equations():
    for r in rows("paired.csv"):
        if r["status"] != "available":
            assert r.get("p_value", "") == r.get("reject", "") == ""
            continue
        x, n = int(r["exceedances"]), int(r["successful_draws"])
        assert n == 99 and int(r["failed_draws"]) == 0
        assert float(r["p_value"]) == (1 + x) / (n + 1)
        assert (r["reject"] == "True") == ((1 + x) / (n + 1) <= 0.05)
        low, high = float(r["tail_lower"]), float(r["tail_upper"])

        def tail(p, lo, hi, n=n):
            return math.fsum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(lo, hi + 1))

        assert low == 0 if x == 0 else tail(low, x, n) == pytest.approx(0.025, abs=1e-10)
        assert high == 1 if x == n else tail(high, 0, x) == pytest.approx(0.025, abs=1e-10)


def test_point_recovery_and_rejection_denominators_are_separate():
    data = rows("paired.csv")
    assert len(data) == 500
    for s in read("paired.json")["summaries"]:
        group = [r for r in data if r["case"] == s["case"] and r["criterion"] == s["criterion"]]
        available = [r for r in group if r["status"] == "available"]
        points = [r for r in group if r.get("selected_k")]
        x = sum(r["reject"] == "True" for r in available)
        assert len(group) == s["requested"] == 25
        assert len(available) == s["available_tests"] and len(points) == s["available_points"]
        assert s["rejected"] == x and s["rejection_wilson_95"] == pytest.approx(
            wilson(x, len(available)), abs=1e-12
        )
        assert s["selected_k_counts"] == {
            str(k): sum(int(r["selected_k"]) == k for r in points) for k in range(1, 5)
        }
        ari = [float(r["ari"]) for r in points if r.get("ari")]
        if ari:
            assert s["ari"]["mean"] == pytest.approx(statistics.mean(ari), abs=1e-12)
            assert s["ari"]["mean_mcse"] == pytest.approx(
                statistics.stdev(ari) / math.sqrt(len(ari)), abs=1e-12
            )
        else:
            assert s["ari"] is None
