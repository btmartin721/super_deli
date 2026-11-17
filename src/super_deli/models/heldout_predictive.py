"""Native one-component replays and individual-first omitted prediction summaries."""

from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from super_deli.io.config import write_json
from super_deli.models.mixture import FIT_LOCK, select_gaussian_mixture


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-9)


def logdensity(scores, parameters):
    """Independent shared-covariance density from retained plain numerical arrays."""
    x = np.asarray(scores)
    w, m, c = (np.asarray(parameters[k]) for k in ("weights", "means", "covariances"))
    if (
        x.ndim != 2
        or m.ndim != 2
        or w.shape != (len(m),)
        or m.shape[1] != x.shape[1]
        or c.shape != (x.shape[1], x.shape[1])
        or any(not np.isfinite(a).all() for a in (x, w, m, c))
        or np.any(w <= 0)
    ):
        raise ValueError("Invalid predictive density parameters")
    close(w.sum(), 1)
    close(c, c.T)
    sign, ld = np.linalg.slogdet(c)
    if sign != 1 or np.linalg.eigvalsh(c).min() <= 0:
        raise ValueError("Predictive covariance must be positive definite")
    terms = []
    for weight, mean in zip(w, m, strict=True):
        delta = x - mean
        q = np.sum(delta * np.linalg.solve(c, delta.T).T, axis=1)
        terms.append(np.log(weight) - 0.5 * (x.shape[1] * np.log(2 * np.pi) + ld + q))
    terms = np.array(terms).T
    total = logsumexp(terms, axis=1)
    if not np.isfinite(total).all():
        raise ValueError("Nonfinite predictive density")
    return total, np.exp(terms - total[:, None]), terms.argmax(1)


def source_arrays(source, record):
    path = source.root / "subsamples" / record["parameters"] / "fit.npz"
    with np.load(path, allow_pickle=False) as f:
        a = {k: f[k].copy() for k in f.files}
    n, d, k = len(source.point.units), source.point.model.n_pcs, record["selected_k"]
    draw = np.asarray(record["training_indices"])
    if (
        a["training_scores"].shape != (len(draw), d)
        or a["projected_scores"].shape != (n, d)
        or a["means"].shape != (k, d)
        or a["training_membership"].shape != (len(draw), k)
        or a["projected_membership"].shape != (n, k)
        or not np.isfinite(a["training_scores"]).all()
    ):
        raise ValueError("Retained source scores/parameters differ from declaration")
    close(a["training_scores"], a["projected_scores"][draw])
    ll, membership, labels = logdensity(a["projected_scores"], a)
    close(membership, a["projected_membership"])
    np.testing.assert_array_equal(labels, a["projected_labels"])
    np.testing.assert_array_equal(labels, record["projected_labels"])
    _, membership, _ = logdensity(a["training_scores"], a)
    close(membership, a["training_membership"])
    return a, ll


def replay_baseline(source, record, arrays):
    settings = replace(source.point.model, max_k=1, n_bootstrap=0)
    fit = select_gaussian_mixture(
        arrays["training_scores"],
        settings,
        record["fit_seed"],
        covariance_type="tied",
        unit_indices=np.arange(len(record["training_indices"])),
    )
    expected = next(c for c in record["candidates"] if c["k"] == 1)
    close(fit.candidates[0]["bic"], expected["bic"])
    obj = fit.mixture
    parameters = dict(weights=obj.weights_, means=obj.means_, covariances=obj.covariances_)
    ll, probabilities, labels = logdensity(arrays["projected_scores"], parameters)
    close(ll, obj.score_samples(arrays["projected_scores"]))
    close(probabilities, 1)
    np.testing.assert_array_equal(labels, 0)
    return parameters, ll, fit.candidates[0]


def gain_values(arrays, chosen, baseline, parameters, selected_k):
    raw_gain = chosen - baseline
    gain = raw_gain.copy()
    if selected_k == 1:
        for key in ("weights", "means", "covariances"):
            close(arrays[key], parameters[key])
        close(raw_gain, 0)
        # Verified identical one-group models have exactly zero gain.
        gain.fill(0)
    return raw_gain, gain


def score_draws(source, directory):
    directory = Path(directory)
    directory.mkdir()
    records = []
    with FIT_LOCK, threadpool_limits(limits=1):
        for r in source.records:
            row = dict(
                replicate=r["replicate"],
                omitted_indices=r["omitted_indices"],
                source_selected_k=r.get("selected_k"),
                status="unavailable_native",
                fit_seed=r["fit_seed"],
                baseline_search_attempted=False,
            )
            if r["status"] == "complete":
                try:
                    a, chosen = source_arrays(source, r)
                    row["baseline_search_attempted"] = True
                    parameters, baseline, candidate = replay_baseline(source, r, a)
                    raw_gain, gain = gain_values(a, chosen, baseline, parameters, r["selected_k"])
                except (
                    ValueError,
                    RuntimeError,
                    AssertionError,
                    np.linalg.LinAlgError,
                    FloatingPointError,
                ) as exc:
                    row.update(
                        status="unavailable_predictive", error=f"{type(exc).__name__}: {exc}"
                    )
                else:
                    # Export errors stop the workflow; they are not statistical draw failures.
                    name = f"draw_{r['replicate']:04d}.npz"
                    with (directory / name).open("xb") as f:
                        np.savez(
                            f,
                            **parameters,
                            selected_logdensity=chosen,
                            baseline_logdensity=baseline,
                            raw_gain=raw_gain,
                            gain=gain,
                        )
                    row.update(status="complete", arrays=name, baseline_candidate=candidate)
            write_json(row, directory / f"draw_{r['replicate']:04d}.json")
            records.append(row)
    source.reverify()
    return records


def summarize_scores(source, records, directory, config):
    n = len(source.point.units)
    requested = np.zeros(n, dtype=np.uint64)
    successful = requested.copy()
    sums = np.zeros(n)
    for row in records:
        absent = np.asarray(row["omitted_indices"], dtype=int)
        requested[absent] += 1
        if row["status"] == "complete":
            with np.load(Path(directory) / row["arrays"], allow_pickle=False) as f:
                gain = f["gain"]
            successful[absent] += 1
            sums[absent] += gain[absent]
    means = np.full(n, np.nan)
    np.divide(sums, successful, out=means, where=successful > 0)
    covered = successful > 0
    coverage = float(covered.mean())
    missing = int(np.sum(requested - successful))
    mean = float(means[covered].mean()) if covered.any() else None
    qualified = coverage >= config.minimum_coverage and missing == 0
    groups = []
    draw_sets = [set(r["training_indices"]) for r in source.records]
    for k in range(source.point.summary["selected_k"]):
        members = np.flatnonzero(source.point.labels == k)
        counts = [len(set(members) & ids) for ids in draw_sets]
        observed = members[covered[members]]
        groups.append(
            dict(
                point_cluster=f"cluster_{k + 1}",
                point_individuals=len(members),
                covered_individuals=len(observed),
                coverage=len(observed) / len(members),
                minimum_included=min(counts),
                maximum_included=max(counts),
                below_declared_minimum=sum(x < source.point.model.min_cluster_size for x in counts),
                covered_individual_mean=None
                if not len(observed)
                else float(means[observed].mean()),
                interpretation=(
                    "Descriptive same-sample point-label annotation; no independent confirmation"
                ),
            )
        )
    eligibility = []
    native_done = [r for r in source.records if r["status"] == "complete"]
    for k in range(1, source.point.model.max_k + 1):
        statuses = [next(c["status"] for c in r["candidates"] if c["k"] == k) for r in native_done]
        eligibility.append(
            dict(
                k=k,
                eligible=statuses.count("eligible"),
                small_component=statuses.count("small_component"),
                insufficient_samples=statuses.count("insufficient_samples"),
                unavailable=len(source.records) - len(native_done),
                selected=sum(r["selected_k"] == k for r in native_done),
            )
        )
    summary = dict(
        status="complete" if missing == 0 else "incomplete",
        candidate_eligibility=eligibility,
        individuals=n,
        requested_draws=len(records),
        successful_draws=sum(r["status"] == "complete" for r in records),
        requested_comparisons=int(requested.sum()),
        successful_comparisons=int(successful.sum()),
        unavailable_requested_comparisons=missing,
        never_omitted_individuals=int(np.sum(requested == 0)),
        covered_individuals=int(covered.sum()),
        coverage=coverage,
        qualified=qualified,
        covered_individual_mean=mean,
        full_cohort_mean=mean if covered.all() and missing == 0 else None,
        direction="unavailable" if not qualified else "positive" if mean > 0 else "nonpositive",
        baseline_searches=sum(r["baseline_search_attempted"] for r in records),
        p_value=None,
        species_probability=None,
        point_analysis_changed=False,
        aggregation="Average omitted gains per individual, then equally over covered individuals",
        missing_gain_bounds="Unavailable: log-density gains have no finite failure bounds",
        scope="Selected versus K=1 in the same frozen projected-score space; no confirmation of K",
    )
    return summary, dict(requested=requested, successful=successful, sums=sums, means=means), groups
