"""Known-law reference classification; never a species or finite-sample ARI bound."""

import numpy as np
from scipy.linalg import block_diag
from scipy.special import logsumexp, ndtr
from sklearn.metrics import adjusted_rand_score

from super_deli.benchmarking.matched_config import SETTINGS
from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.view_signal_metrics import linear_recovery, scatter_fraction
from super_deli.models.separate_neural import unpack_branch


def loading():
    parts = []
    for role in ("morphology", "ecology"):
        a = np.array(SETTINGS["loading_" + role], dtype=float)
        parts.append(a / np.linalg.norm(a, axis=1)[:, None])
    return block_diag(*parts)


def latent_law(task):
    if task["case"] == "gaussian_independent":
        return np.zeros((1, 4)), np.eye(4)
    require(task["case"] == "four_complementary", "Supported known law")
    a = 3 * task["signal_multiplier"]
    return np.array([[-a, 0, -a, 0], [-a, 0, a, 0], [a, 0, -a, 0], [a, 0, a, 0]])[
        :, [0, 1, 3, 2]
    ], np.eye(4)


def affine_pca(point):
    maps, offsets = [], []
    for role in ("morphology", "ecology"):
        a = unpack_branch(point, role)
        m = np.zeros((2, len(a["means"])))
        m[:, a["kept_indices"]] = (
            a["components"] / a["scales"][a["kept_indices"]] / a["latent_scales"][:, None]
        )
        offset = (
            -m @ a["means"]
            - (a["components"] @ a["pca_mean"] + a["latent_means"]) / a["latent_scales"]
        )
        maps.append(m)
        offsets.append(offset)
    return block_diag(*maps), np.concatenate(offsets)


def laws(task, pca_point):
    mean, covariance = latent_law(task)
    loadings = loading()
    raw_mean = mean @ loadings.T
    raw_cov = loadings @ covariance @ loadings.T + 0.15**2 * np.eye(12)
    a, b = affine_pca(pca_point)
    return {
        "latent": (mean, covariance),
        "raw": (raw_mean, raw_cov),
        "pca4": (raw_mean @ a.T + b, a @ raw_cov @ a.T),
    }


def classify(x, means, covariance):
    chol = np.linalg.cholesky(covariance)
    delta = x[:, None, :] - means[None, :, :]
    solved = np.linalg.solve(chol, delta.reshape(-1, x.shape[1]).T).T.reshape(delta.shape)
    logp = -0.5 * (
        (solved**2).sum(2) + x.shape[1] * np.log(2 * np.pi) + 2 * np.log(np.diag(chol)).sum()
    ) - np.log(len(means))
    normalizer = logsumexp(logp, axis=1)
    membership = np.exp(logp - normalizer[:, None])
    require(np.isfinite(membership).all(), "Finite known-law reference")
    return membership, membership.argmax(1)


def expected_accuracy(means, covariance):
    if len(means) == 1:
        return 1.0
    # Independent complementary view blocks remain separate in all three spaces.
    d = 4 if means.shape[1] == 12 else 2
    result = 1.0
    for rows, sl in (((0, 2), slice(0, d)), ((0, 1), slice(d, means.shape[1]))):
        difference = means[rows[1], sl] - means[rows[0], sl]
        sigma = covariance[sl, sl]
        result *= ndtr(0.5 * np.sqrt(difference @ np.linalg.solve(sigma, difference)))
    return float(result)


def metrics(truth, labels):
    ari = float(adjusted_rand_score(truth, labels))
    return dict(
        ari=ari,
        selected_k=len(np.unique(labels)),
        recovered=bool(len(np.unique(labels)) == 4 and ari >= 0.8),
    )


def reference_result(x, truth, mean, covariance):
    membership, labels = classify(x, mean, covariance)
    return dict(
        **metrics(truth, labels),
        accuracy=float(np.mean(truth == labels)),
        expected_fixed_map_accuracy=expected_accuracy(mean, covariance),
        expected_fixed_map_error=1 - expected_accuracy(mean, covariance),
        mean_max_probability=float(membership.max(1).mean()),
        fraction_max_probability_below_08=float(np.mean(membership.max(1) < 0.8)),
        interpretation="Known-law pointwise reference; not an ARI ceiling",
    ), dict(means=mean, covariance=covariance, membership=membership, labels=labels)


def geometry(x, data):
    target = np.column_stack((data["latent_morphology"], data["latent_ecology"]))
    truth = data["truth"]
    return dict(
        rank=int(np.linalg.matrix_rank(x)),
        linear_recovery=linear_recovery(x, target),
        between_scatter_fraction=scatter_fraction(x, truth),
        interpretation="In-sample geometry; no nonlinear information bound",
    )
