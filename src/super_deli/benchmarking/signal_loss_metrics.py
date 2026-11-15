"""Post-fit geometry and within-space criteria; labels never select representations."""

import numpy as np
from sklearn.metrics import adjusted_rand_score

from super_deli.benchmarking.view_signal_metrics import linear_recovery, scatter_fraction


def axes(labels):
    labels = np.asarray(labels, dtype=int)
    a, b = 2 * (labels // 2) - 1, 2 * (labels % 2) - 1
    return np.column_stack((a, b, a * b))


def centroid_matrix(x, labels):
    x = np.asarray(x, dtype=float)
    centered = x - x.mean(axis=0)
    return np.stack(
        [
            np.sqrt(np.sum(labels == g)) * centered[labels == g].mean(axis=0)
            for g in np.unique(labels)
        ]
    )


def projected_energy(x, labels, components):
    between = centroid_matrix(x, labels)
    denominator = np.sum(between**2)
    return float(np.sum((between @ components.T) ** 2) / denominator) if denominator else None


def geometry(x, labels):
    return dict(
        dimensions=x.shape[1],
        scatter_fraction=scatter_fraction(x, labels),
        latent_contrasts=linear_recovery(x, axes(labels)),
        interpretation="post-fit descriptive in-sample geometry; no information bound",
    )


def recovery(labels, predicted, law):
    if law in {"continuous_control", "conflicting"}:
        return dict(
            target_k=None,
            ari=None,
            correct_recovery=None,
            control_split=bool(len(np.unique(predicted)) > 1)
            if law == "continuous_control"
            else None,
        )
    target = labels // 2 if law == "genomic_only" else labels
    ari = float(adjusted_rand_score(target, predicted))
    k = len(np.unique(target))
    return dict(
        target_k=k,
        ari=ari,
        correct_recovery=bool(len(np.unique(predicted)) == k and ari >= 0.8),
        control_split=None,
    )


def parameters(k, d, covariance):
    return (
        k
        - 1
        + k * d
        + ({"diag": k * d, "full": k * d * (d + 1) // 2, "tied": d * (d + 1) // 2}[covariance])
    )


def costs(candidates, n, d, covariance):
    result = []
    for c in candidates:
        if c.get("bic") is None:
            continue
        penalty = parameters(c["k"], d, covariance) * np.log(n)
        result.append(
            dict(
                k=c["k"],
                status=c["status"],
                bic=c["bic"],
                penalty=float(penalty),
                log_likelihood=float((penalty - c["bic"]) / 2),
                likelihood_origin="reconstructed from retained native BIC",
            )
        )
    k1 = next((r for r in result if r["k"] == 1), None)
    if k1:
        for r in result:
            r.update(
                twice_likelihood_gain=2 * (r["log_likelihood"] - k1["log_likelihood"]),
                extra_penalty=r["penalty"] - k1["penalty"],
                delta_bic=r["bic"] - k1["bic"],
            )
    return result
