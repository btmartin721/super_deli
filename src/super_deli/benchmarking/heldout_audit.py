"""Independent densities, native baseline moments and exposure-weighted aggregation."""

from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from super_deli.benchmarking.full_retention_audit import close
from super_deli.io.config import read_json, write_json
from super_deli.io.heldout_support_source import load_subsample_source


def density(x, parameters):
    terms = np.column_stack(
        [
            np.log(w) + multivariate_normal.logpdf(x, mean=m, cov=parameters["covariances"])
            for w, m in zip(parameters["weights"], parameters["means"], strict=True)
        ]
    )
    return logsumexp(terms, axis=1)


def audit_report(directory, *, write=True, source=None):
    directory = Path(directory)
    if source is None:
        source = load_subsample_source(directory / "reference")
    binding = read_json(directory / "source.json")
    assert binding["files"] == source.files
    config = read_json(directory / "configuration.json")
    assert config == dict(schema_version=1, policy="native_k1_omitted_v1", minimum_coverage=0.95)
    records = read_json(directory / "predictive_replicates.json")
    assert len(records) == len(source.records)
    n = len(source.point.units)
    values = [[] for _ in range(n)]
    attempts = [0] * n
    checks = 0
    baseline_attempts = 0
    successful = 0
    for r, original in zip(records, source.records, strict=True):
        assert (
            r["replicate"] == original["replicate"]
            and r["omitted_indices"] == original["omitted_indices"]
        )
        assert r["fit_seed"] == original["fit_seed"]
        baseline_attempts += r["baseline_search_attempted"]
        absent = original["omitted_indices"]
        for i in absent:
            attempts[i] += 1
        if r["status"] != "complete":
            assert "arrays" not in r
            continue
        assert original["status"] == "complete"
        successful += 1
        with np.load(
            source.root / "subsamples" / original["parameters"] / "fit.npz", allow_pickle=False
        ) as a:
            with np.load(directory / "scores" / r["arrays"], allow_pickle=False) as b:
                x = a["training_scores"]
                full = a["projected_scores"]
                mean = x.mean(axis=0)
                centered = x - mean
                covariance = centered.T @ centered / len(x) + source.point.model.reg_covar * np.eye(
                    x.shape[1]
                )
                close(b["weights"], [1.0])
                close(b["means"], mean[None])
                close(b["covariances"], covariance)
                base = density(full, b)
                selected = density(full, a)
                close(base, b["baseline_logdensity"])
                close(selected, b["selected_logdensity"])
                raw = selected - base
                close(raw, b["raw_gain"])
                if original["selected_k"] == 1:
                    close(raw, 0)
                    gain = np.zeros(n)
                else:
                    gain = raw
                close(gain, b["gain"])
                train = density(x, b)
                d = x.shape[1]
                bic = -2 * train.sum() + (d + d * (d + 1) // 2) * np.log(len(x))
                expected = next(c for c in original["candidates"] if c["k"] == 1)
                close(bic, expected["bic"])
                close(r["baseline_candidate"]["bic"], bic)
                for i in absent:
                    values[i].append(float(gain[i]))
                checks += 2 * n + len(x)
    means = [float(np.mean(v)) if v else None for v in values]
    covered = [i for i, v in enumerate(values) if v]
    gain = float(np.mean([means[i] for i in covered])) if covered else None
    missing = sum(attempts) - sum(map(len, values))
    coverage = len(covered) / n
    summary = read_json(directory / "summary.json")
    assert (
        summary["baseline_searches"] == baseline_attempts
        and summary["successful_draws"] == successful
    )
    assert summary["unavailable_requested_comparisons"] == missing
    assert summary["requested_comparisons"] == sum(attempts)
    assert summary["successful_comparisons"] == sum(map(len, values))
    assert summary["covered_individuals"] == len(covered)
    close(summary["coverage"], coverage)
    assert summary["qualified"] == (coverage >= 0.95 and missing == 0)
    assert summary["p_value"] is None and summary["point_analysis_changed"] is False
    if gain is None:
        assert summary["covered_individual_mean"] is None
    else:
        close(summary["covered_individual_mean"], gain)
    if len(covered) == n and missing == 0:
        close(summary["full_cohort_mean"], gain)
    else:
        assert summary["full_cohort_mean"] is None
    expected_direction = (
        ("positive" if gain > 0 else "nonpositive") if summary["qualified"] else "unavailable"
    )
    assert summary["direction"] == expected_direction
    with np.load(directory / "individual_scores.npz", allow_pickle=False) as a:
        np.testing.assert_array_equal(a["requested"], attempts)
        np.testing.assert_array_equal(a["successful"], [len(v) for v in values])
        close(a["sums"], [sum(v) for v in values])
        close(a["means"], [np.nan if m is None else m for m in means])
    for k, g in enumerate(read_json(directory / "cluster_support.json")):
        members = np.flatnonzero(source.point.labels == k)
        counts = [len(set(members) & set(r["training_indices"])) for r in source.records]
        available = [i for i in members if values[i]]
        assert g["point_individuals"] == len(members) and g["covered_individuals"] == len(available)
        assert g["minimum_included"] == min(counts) and g["maximum_included"] == max(counts)
        assert g["below_declared_minimum"] == sum(
            c < source.point.model.min_cluster_size for c in counts
        )
        if available:
            close(g["covered_individual_mean"], np.mean([means[i] for i in available]))
        else:
            assert g["covered_individual_mean"] is None
    for row in summary["candidate_eligibility"]:
        done = [r for r in source.records if r["status"] == "complete"]
        states = [next(c["status"] for c in r["candidates"] if c["k"] == row["k"]) for r in done]
        assert row["eligible"] == states.count("eligible")
        assert row["small_component"] == states.count("small_component")
        assert row["insufficient_samples"] == states.count("insufficient_samples")
        assert row["unavailable"] == len(source.records) - len(done)
        assert row["selected"] == sum(r["selected_k"] == row["k"] for r in done)
    source.reverify()
    result = dict(
        passed=True,
        draws=len(records),
        baseline_searches=baseline_attempts,
        successful=successful,
        density_checks=checks,
        new_fits=0,
        point_changed=False,
    )
    if write:
        write_json(result, directory / "independent-audit.json")
    return result
