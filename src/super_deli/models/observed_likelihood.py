"""Observed-coordinate densities and exact conditional-moment EM updates.

The covariance M-step maximizes Q subject to Sigma >= bound * I. It does
not add ridge variance to a fitted covariance. No source values are imputed.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp


@dataclass(frozen=True)
class Parameters:
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray


@dataclass(frozen=True)
class Pattern:
    rows: np.ndarray
    observed: np.ndarray
    missing: np.ndarray
    values: np.ndarray


def patterns(x):
    masks, inverse = np.unique(~np.isnan(x), axis=0, return_inverse=True)
    result = []
    for i, mask in enumerate(masks):
        if not mask.any():
            raise ValueError("All-missing rows have no likelihood information")
        rows = np.flatnonzero(inverse == i)
        observed, missing = np.flatnonzero(mask), np.flatnonzero(~mask)
        result.append(Pattern(rows, observed, missing, x[np.ix_(rows, observed)]))
    return tuple(result)


def constrain_covariance(scatter, bound):
    scatter = (scatter + scatter.T) / 2
    eigenvalues, vectors = np.linalg.eigh(scatter)
    # Large negative scatter eigenvalues indicate failed arithmetic, not a bound.
    if eigenvalues[0] < -1e-8 * max(1.0, float(np.abs(scatter).max())):
        raise FloatingPointError("Conditional scatter is not positive semidefinite")
    return (vectors * np.maximum(eigenvalues, bound)) @ vectors.T


def evaluate(n, parameters, groups):
    k = len(parameters.weights)
    joint = np.empty((n, k))
    if np.any(parameters.weights <= 0) or not np.isclose(parameters.weights.sum(), 1.0):
        raise FloatingPointError("Invalid mixture weights")
    for group in groups:
        o = group.observed
        for c in range(k):
            cov = parameters.covariances[c][np.ix_(o, o)]
            factor = cho_factor(cov, lower=True, check_finite=False)
            residual = group.values - parameters.means[c, o]
            quadratic = np.sum(
                residual * cho_solve(factor, residual.T, check_finite=False).T, axis=1
            )
            logdet = 2 * np.log(np.diag(factor[0])).sum()
            joint[group.rows, c] = np.log(parameters.weights[c]) - 0.5 * (
                len(o) * np.log(2 * np.pi) + logdet + quadratic
            )
    density = logsumexp(joint, axis=1)
    membership = np.exp(joint - density[:, None])
    if not np.isfinite(density).all() or not np.isfinite(membership).all():
        raise FloatingPointError("Nonfinite observed likelihood")
    return density, membership


def conditional_offsets(group, mean, covariance):
    """E[X-mean | observed], and Cov[X | observed], for one component/pattern."""
    o, m = group.observed, group.missing
    delta = group.values - mean[o]
    offsets = np.zeros((len(group.rows), len(mean)))
    offsets[:, o] = delta
    conditional = np.zeros_like(covariance)
    if len(m):
        factor = cho_factor(covariance[np.ix_(o, o)], lower=True, check_finite=False)
        coefficient = cho_solve(factor, covariance[np.ix_(o, m)], check_finite=False)
        offsets[:, m] = delta @ coefficient
        block = covariance[np.ix_(m, m)] - covariance[np.ix_(m, o)] @ coefficient
        conditional[np.ix_(m, m)] = (block + block.T) / 2
    return offsets, conditional


def update(n, parameters, groups, bound):
    """An exact constrained M-step using conditional first and second moments."""
    _, responsibilities = evaluate(n, parameters, groups)
    mass = responsibilities.sum(axis=0)
    if np.any(mass <= np.finfo(float).eps * n):
        raise FloatingPointError("Component responsibility mass is numerically zero")
    means, covariances = [], []
    for c, old_mean in enumerate(parameters.means):
        first = np.zeros_like(old_mean)
        second = np.zeros_like(parameters.covariances[c])
        for group in groups:
            offsets, conditional = conditional_offsets(group, old_mean, parameters.covariances[c])
            r = responsibilities[group.rows, c]
            first += r @ offsets
            second += (offsets.T * r) @ offsets + r.sum() * conditional
        shift = first / mass[c]
        means.append(old_mean + shift)
        scatter = second / mass[c] - np.outer(shift, shift)
        covariances.append(constrain_covariance(scatter, bound))
    return Parameters(mass / n, np.array(means), np.array(covariances))


def parameter_change(a, b):
    return max(
        float(np.max(np.abs(x - y) / np.maximum(1.0, np.maximum(np.abs(x), np.abs(y)))))
        for x, y in zip(
            (a.weights, a.means, a.covariances), (b.weights, b.means, b.covariances), strict=True
        )
    )
