"""Candidate fit/penalty, covariance, count association and held-out evidence."""

from __future__ import annotations

import io


def densities(scores, weights, means, covariance):
    import numpy as np
    from scipy.special import logsumexp

    columns = []
    for w, mean, cov in zip(weights, means, covariance, strict=True):
        factor = np.linalg.cholesky(cov)
        z = np.linalg.solve(factor, (scores - mean).T).T
        columns.append(
            np.log(w)
            - 0.5
            * (
                scores.shape[1] * np.log(2 * np.pi)
                + 2 * np.log(np.diag(factor)).sum()
                + (z * z).sum(1)
            )
        )
    return logsumexp(np.column_stack(columns), axis=1)


def candidate_metrics(raw_pool, candidates, *, test_scores=None, counts=None, reg_covar=1e-6):
    import numpy as np
    from sklearn.metrics import mutual_info_score

    pool = np.load(io.BytesIO(raw_pool), allow_pickle=False)
    with pool:
        scores = pool["scores"]
        n, d = scores.shape
        one = next(c for c in candidates if c["k"] == 1)
        ll1 = one["log_likelihood"]
        delta = scores - pool["k1_means"][0]
        distance = np.sum(delta * np.linalg.solve(pool["k1_covariances"][0], delta.T).T, axis=1)
        out, held = [], {}
        for c in candidates:
            if c["status"] == "insufficient_samples":
                out.append(dict(c))
                continue
            k = c["k"]
            weights, means, cov, membership, labels = (
                pool[f"k{k}_{name}"]
                for name in ("weights", "means", "covariances", "membership", "labels")
            )
            q = k - 1 + k * d + k * d * (d + 1) // 2
            eigen = np.linalg.eigvalsh(cov)
            row = {
                **c,
                "parameters": q,
                "parameters_per_unit": q / n,
                "twice_loglikelihood_gain_over_one": 2 * (c["log_likelihood"] - ll1),
                "aic_penalty_over_one": 2 * (q - (d + d * (d + 1) // 2)),
                "bic_penalty_over_one": float(np.log(n) * (q - (d + d * (d + 1) // 2))),
                "minimum_eigenvalues": eigen[:, 0].tolist(),
                "minimum_eigenvalue_over_regularizer": float(eigen.min() / reg_covar),
                "condition_numbers": (eigen[:, -1] / eigen[:, 0]).tolist(),
                "weights": weights.tolist(),
                "component_mean_mahalanobis_squared": [
                    float(distance[labels == j].mean()) if np.any(labels == j) else None
                    for j in range(k)
                ],
                "mean_assignment_entropy": float(
                    -np.sum(membership * np.log(np.maximum(membership, 1e-300))) / n
                ),
            }
            if counts is not None:
                table = [
                    [int(np.sum((counts == m) & (labels == a))) for a in range(k)]
                    for m in range(1, 5)
                ]
                row.update(
                    count_by_component=table,
                    count_label_mutual_information=float(mutual_info_score(counts, labels)),
                )
            if test_scores is not None:
                held[str(k)] = densities(test_scores, weights, means, cov)
            out.append(row)
        if held:
            for row in out:
                if str(row["k"]) in held:
                    gain = held[str(row["k"])] - held["1"]
                    row["heldout_mean_logdensity_gain_over_one"] = float(gain.mean())
                    row["heldout_mcse_conditional_on_training"] = float(
                        gain.std(ddof=1) / np.sqrt(len(gain))
                    )
        eligible = [c for c in candidates if c["status"] == "eligible"]
        selected = {
            kind + "_k": min(eligible, key=lambda c: (c[kind], c["k"]))["k"]
            for kind in ("aic", "bic")
        }
        return {
            "n": n,
            "dimension": d,
            **selected,
            "candidates": out,
            "one_component_mahalanobis_squared": distance.tolist(),
            "extreme_unit_indices": np.argsort(-distance, kind="stable")[: min(10, n)].tolist(),
        }, held
