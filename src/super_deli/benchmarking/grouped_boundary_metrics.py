"""Descriptive boundary, sampling-precision and component diagnostics."""

import io

import numpy as np
from scipy.linalg import eigvalsh

from super_deli.benchmarking.full_grouped_diagnostic_analysis import describe
from super_deli.benchmarking.recovery_analysis import _wilson


def conditional(covariance):
    regression = np.linalg.solve(covariance[4:, 4:], covariance[4:, :4]).T
    return covariance[:4, :4] - regression @ covariance[4:, :4]


def raw_metrics(fitted, truth):
    c, t = conditional(fitted.covariance), conditional(truth.covariance)
    values, vectors = np.linalg.eigh(truth.within)
    whiten = (vectors / np.sqrt(values)) @ vectors.T

    def contrast(between, within):
        traces = [np.trace(whiten @ (between + within / n) @ whiten) for n in (1, 4)]
        return float(traces[0] / traces[1])

    return {
        "fitted_rank": fitted.layout["diagnostics"]["conditional_rank"],
        "true_rank": int(np.sum(eigvalsh(t, truth.within) > 1e-8)),
        "conditional_error": float(np.linalg.norm(whiten @ (c - t) @ whiten)),
        "within_relative_error": float(np.linalg.norm(whiten @ fitted.within @ whiten - np.eye(4))),
        "true_conditional_trace": float(np.trace(whiten @ t @ whiten)),
        "fitted_conditional_trace": float(np.trace(whiten @ c @ whiten)),
        "true_count_variance_ratio": contrast(t, truth.within),
        "fitted_count_variance_ratio": contrast(c, fitted.within),
        "optimality": fitted.layout["diagnostics"]["optimality"]["maximum"],
        "minimum_count": int(min(fitted.counts)),
        "maximum_count": int(max(fitted.counts)),
        "count_frequencies": {str(n): int(sum(fitted.counts == n)) for n in range(1, 5)},
        "true_conditional_eigenvalues": eigvalsh(t, truth.within).tolist(),
        "fitted_conditional_eigenvalues": eigvalsh(c, truth.within).tolist(),
    }


def count_association(labels, counts):
    """Cramer's V as a descriptive association; no post-selection significance claim."""
    a, b = np.unique(labels, return_inverse=True)[1], np.unique(counts, return_inverse=True)[1]
    table = np.zeros((max(a) + 1, max(b) + 1))
    np.add.at(table, (a, b), 1)
    if min(table.shape) == 1:
        return 0.0
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    return float(
        np.sqrt(np.sum((table - expected) ** 2 / expected) / (table.sum() * (min(table.shape) - 1)))
    )


def point_metrics(archive, record, counts, model):
    if record["status"] != "complete":
        return {"status": "failed"}
    prefix = record["archive_prefix"] + "/run/"
    eligible = [c for c in record["candidates"] if c["status"] == "eligible"]
    best = max(eligible, key=lambda c: (c["log_likelihood"], -c["k"]))
    with np.load(
        io.BytesIO(archive.read(prefix + "gmm_candidates.npz")), allow_pickle=False
    ) as pool:
        minimum = min(
            np.linalg.eigvalsh(pool[f"k{c['k']}_covariances"]).min()
            for c in record["candidates"]
            if c["status"] != "insufficient_samples"
        )
        result = {
            "count_association": count_association(pool[f"k{record['selected_k']}_labels"], counts),
            "lr_count_association": count_association(pool[f"k{best['k']}_labels"], counts),
            "covariance_min_over_regularizer": float(minimum / model.reg_covar),
        }
    return {
        "status": "complete",
        "statistic": record["statistic"],
        "bic_statistic": record["bic_statistic"],
        "selected_k": record["selected_k"],
        "bic_selected_k": record["bic_selected_k"],
        "lr_maximizer_k": best["k"],
        "eligible_ks": [c["k"] for c in eligible],
        "maximum_iterations": max(c.get("iterations", 0) for c in record["candidates"]),
        **retained_variance(archive, prefix, model),
        **result,
    }


def summaries(rows, covariance, pipelines):
    output, pairs = [], []
    keys = (
        "statistic",
        "count_association",
        "lr_count_association",
        "covariance_min_over_regularizer",
        "morphology_retained_variance",
        "ecology_retained_variance",
    )
    for cohort, case, arm, statistic in sorted(
        {(r["cohort"], r["case"], r["arm"], r["statistic_kind"]) for r in rows}
    ):
        group = [
            r
            for r in rows
            if (r["cohort"], r["case"], r["arm"], r["statistic_kind"])
            == (cohort, case, arm, statistic)
        ]
        available = [r for r in group if r["p_value"] is not None]
        events = sum(r["reject"] for r in available)
        relevant = [
            r
            for r in pipelines
            if r["cohort"] == cohort
            and r["case"] == case
            and r["arm"] == arm
            and r.get("replicate", -1) >= 0
            and r["status"] == "complete"
        ]
        cov = [r for r in covariance if r["cohort"] == cohort and r["case"] == case]
        output.append(
            {
                "cohort": cohort,
                "case": case,
                "arm": arm,
                "statistic_kind": statistic,
                "requested": len(group),
                "available": len(available),
                "rejected": events,
                "wilson_95": _wilson(events, len(available)),
                "withheld_bounds": [
                    events / len(group),
                    (events + len(group) - len(available)) / len(group),
                ],
                "null_pipeline_metrics": {
                    key: describe([r.get(key) for r in relevant]) for key in keys
                },
                "raw_metrics": {
                    key: describe([r.get(key) for r in cov])
                    for key in (
                        "conditional_error",
                        "fitted_conditional_trace",
                        "true_conditional_trace",
                        "fitted_count_variance_ratio",
                        "true_count_variance_ratio",
                        "optimality",
                    )
                },
                "fitted_ranks": {
                    str(k): sum(r.get("fitted_rank") == k for r in cov) for k in range(5)
                },
            }
        )
    for case in sorted({r["case"] for r in rows}):
        for statistic in ("likelihood", "bic"):
            left = {
                r["repetition"]: r
                for r in rows
                if r["cohort"] == "fresh"
                and r["case"] == case
                and r["arm"] == "fitted"
                and r["statistic_kind"] == statistic
            }
            right = {
                r["repetition"]: r
                for r in rows
                if r["cohort"] == "fresh"
                and r["case"] == case
                and r["arm"] == "known"
                and r["statistic_kind"] == statistic
            }
            ids = [
                i
                for i in left
                if left[i]["p_value"] is not None and right[i]["p_value"] is not None
            ]
            pairs.append(
                {
                    "case": case,
                    "statistic_kind": statistic,
                    "available_pairs": len(ids),
                    "fitted_only_reject": sum(
                        left[i]["reject"] and not right[i]["reject"] for i in ids
                    ),
                    "known_only_reject": sum(
                        right[i]["reject"] and not left[i]["reject"] for i in ids
                    ),
                    "p_fitted_minus_known": describe(
                        [left[i]["p_value"] - right[i]["p_value"] for i in ids]
                    ),
                }
            )
    return {"summaries": output, "paired_changes": pairs}


def retained_variance(archive, prefix, model):
    result = {}
    for i, view in enumerate(model.views):
        with np.load(
            io.BytesIO(archive.read(prefix + f"view_{i:03d}/parameters/pca.npz")),
            allow_pickle=False,
        ) as pca:
            key = "morphology" if view.kind == "morphological" else "ecology"
            result[key + "_retained_variance"] = float(pca["explained_variance_ratio"].sum())
    return result
