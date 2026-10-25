"""Label-invariant stability and finite-draw precision, with explicit failure bounds."""

from collections import Counter


def frequencies(counts, requested, successful, *, minimum=1):
    import numpy as np
    from scipy.stats import beta

    from super_deli.models.unit_bootstrap import _bounds

    counts, requested, successful = np.broadcast_arrays(counts, requested, successful)
    low, high, frequency = _bounds(counts, requested, successful)
    coverage = successful >= minimum
    frequency = np.where(coverage, frequency, np.nan)
    conditional = np.divide(counts, successful, out=np.full(counts.shape, np.nan), where=coverage)
    mcse = np.sqrt(conditional * (1 - conditional) / np.maximum(successful, 1))
    ci_low = np.where(counts == 0, 0, beta.ppf(0.025, counts, successful - counts + 1))
    ci_high = np.where(counts == successful, 1, beta.ppf(0.975, counts + 1, successful - counts))
    return {
        "counts": counts,
        "requested": requested,
        "successful": successful,
        "frequency": frequency,
        "failure_lower": low,
        "failure_upper": high,
        "conditional_frequency": conditional,
        "conditional_mcse": mcse,
        "conditional_mc_low": np.where(coverage, ci_low, np.nan),
        "conditional_mc_high": np.where(coverage, ci_high, np.nan),
    }


def summarize_partitions(point_labels, point_k, records):
    import numpy as np

    labels = np.asarray(point_labels)
    n, requested = len(labels), len(records)
    counts = np.zeros((n, n), dtype=np.int64)
    oob_counts, oob_requested, oob_success = (np.zeros_like(counts) for _ in range(3))
    groups = [set(np.flatnonzero(labels == k)) for k in range(point_k)]
    exact = np.zeros(point_k, dtype=np.int64)
    selected = Counter()
    success = 0
    for row in records:
        draw = row["training_indices"]
        if len(draw) != n or any(type(i) is not int or not 0 <= i < n for i in draw):
            raise ValueError("Bootstrap draw must contain N original locality indices")
        oob = np.bincount(draw, minlength=n) == 0
        pairs = np.logical_and.outer(oob, oob)
        oob_requested += pairs
        if row["status"] != "complete":
            continue
        k, prediction = row["selected_k"], np.asarray(row["labels"])
        if (
            type(k) is not int
            or k < 1
            or prediction.shape != (n,)
            or prediction.dtype.kind not in "iu"
            or np.any(prediction < 0)
            or np.any(prediction >= k)
        ):
            raise ValueError("Successful records require a valid original-unit partition")
        same = np.equal.outer(prediction, prediction)
        counts += same
        oob_counts += same & pairs
        oob_success += pairs
        clusters = [set(np.flatnonzero(prediction == j)) for j in range(k)]
        exact += [g in clusters for g in groups]
        selected[k] += 1
        success += 1
    arrays = {}
    for prefix, numerator, denominator, completed, minimum in (
        ("all", counts, requested, success, 1),
        ("oob", oob_counts, oob_requested, oob_success, 10),
        ("exact_cluster", exact, requested, success, 1),
        (
            "selected_k",
            np.asarray([selected[k] for k in range(1, max([point_k, *selected]) + 1)]),
            requested,
            success,
            1,
        ),
    ):
        arrays.update(
            {
                prefix + "_" + key: value
                for key, value in frequencies(
                    numerator, denominator, completed, minimum=minimum
                ).items()
            }
        )

    def finite_mean(a, mask):
        values = a[mask]
        return float(np.nanmean(values)) if np.isfinite(values).any() else None

    clusters = []
    for j, group in enumerate(groups):
        inside = np.zeros(n, bool)
        inside[list(group)] = True
        within = np.logical_and.outer(inside, inside) & ~np.eye(n, dtype=bool)
        cross = np.logical_and.outer(inside, ~inside)
        row = {
            "point_cluster": j,
            "unit_indices": sorted(map(int, group)),
            "exact_recovery_count": int(exact[j]),
        }
        for prefix in ("all", "oob"):
            for name, mask in (("within_cohesion", within), ("cross_merging", cross)):
                row[prefix + "_" + name] = finite_mean(arrays[prefix + "_frequency"], mask)
                row[prefix + "_" + name + "_conditional"] = finite_mean(
                    arrays[prefix + "_conditional_frequency"], mask
                )
                row[prefix + "_" + name + "_covered_pairs"] = int(
                    np.sum(np.isfinite(arrays[prefix + "_conditional_frequency"]) & mask)
                )
                row[prefix + "_" + name + "_total_pairs"] = int(mask.sum())
        clusters.append(row)
    return {
        "status": "available" if success == requested else "unavailable",
        "requested": requested,
        "successful": success,
        "failed": requested - success,
        "selected_k_counts": dict(sorted(selected.items())),
        "clusters": clusters,
        "oob_minimum_successful_pairs": 10,
        "scope": "Conditional resampling stability; no species probability or significance",
        "mc_intervals": "Unadjusted exact binomial precision for conditional draw events; "
        "no biological or true-K confidence coverage",
    }, arrays
