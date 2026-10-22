"""Profiled marginal ML for unequal Gaussian random-intercept groups.

Group means have covariance B + diag(R/n_g, 0); within-morphology
contrasts identify R. Ecology is observed once per group. No ridge or
covariance clipping is applied to accepted fits.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def sufficient_statistics(morphology, ecology, targets):
    counts = np.bincount(targets, minlength=len(ecology))
    if np.any(counts == 0):
        raise ValueError("Every ecology group requires morphology observations")
    means = np.stack([morphology[targets == g].mean(axis=0) for g in range(len(ecology))])
    centered = morphology - means[targets]
    return np.column_stack((means, ecology)), counts, centered.T @ centered


def _factor(vector, width):
    lower = np.zeros((width, width))
    lower[np.tril_indices(width)] = vector
    lower[np.diag_indices(width)] = np.exp(np.diag(lower))
    return lower


def pack_covariance(covariance):
    lower = np.linalg.cholesky(covariance)
    lower[np.diag_indices(len(lower))] = np.log(np.diag(lower))
    return lower[np.tril_indices(len(lower))]


def profile_likelihood(parameters, z, counts, scatter, *, constant=False):
    """Return negative log likelihood, analytic gradient and profiled mean.

    The optional constant makes the value equal the dense raw-data Gaussian
    likelihood, including the group-mean transformation Jacobian.
    """
    g, d = z.shape
    p = len(scatter)
    split = d * (d + 1) // 2
    lb, lr = _factor(parameters[:split], d), _factor(parameters[split:], p)
    between, within = lb @ lb.T, lr @ lr.T
    unique = np.unique(counts)
    inverses, logdets = {}, {}
    precision, weighted = np.zeros((d, d)), np.zeros(d)
    for n in unique:
        covariance = between.copy()
        covariance[:p, :p] += within / n
        inverse = np.linalg.inv(covariance)
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0 or not np.isfinite(logdet):
            raise ValueError("Nonfinite group covariance during null fitting")
        rows = z[counts == n]
        precision += len(rows) * inverse
        weighted += inverse @ rows.sum(axis=0)
        inverses[n], logdets[n] = inverse, logdet
    mean = np.linalg.solve(precision, weighted)
    objective, gb, gr = 0.0, np.zeros((d, d)), np.zeros((p, p))
    for n in unique:
        rows = z[counts == n] - mean
        inverse = inverses[n]
        projected = rows @ inverse
        objective += 0.5 * (len(rows) * logdets[n] + np.sum(projected * rows))
        gradient = 0.5 * (len(rows) * inverse - projected.T @ projected)
        gb += gradient
        gr += gradient[:p, :p] / n
    residual_df = int(counts.sum()) - g
    inverse = np.linalg.inv(within)
    objective += 0.5 * (residual_df * np.linalg.slogdet(within)[1] + np.trace(inverse @ scatter))
    gr += 0.5 * (residual_df * inverse - inverse @ scatter @ inverse)
    gradients = []
    for grad, factor in ((gb, lb), (gr, lr)):
        transformed = 2 * grad @ factor
        transformed[np.diag_indices(len(factor))] *= np.diag(factor)
        gradients.append(transformed[np.tril_indices(len(factor))])
    if constant:
        objective += 0.5 * (
            p * np.log(counts).sum() + (counts.sum() * p + g * (d - p)) * np.log(2 * np.pi)
        )
    return objective, np.concatenate(gradients), mean


def check_rank(covariance, tolerance, name):
    eigenvalues = np.linalg.eigvalsh(covariance)
    if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= tolerance * max(
        float(eigenvalues[-1]), 1.0
    ):
        raise ValueError(f"{name} covariance is singular or on the unsupported boundary")


def fit_gaussian(morphology, ecology, targets, config):
    z, counts, scatter = sufficient_statistics(morphology, ecology, targets)
    g, d = z.shape
    p, residual_df = morphology.shape[1], len(morphology) - g
    if g <= d or g < 2 * config.model.min_cluster_size or residual_df < p:
        raise ValueError(
            "Require G>P+Q, two feasible components and residual degrees of freedom >=P"
        )
    initial_mean = z.mean(axis=0)
    covariance = (z - initial_mean).T @ (z - initial_mean) / g
    pooled = scatter / residual_df
    scales = np.sqrt(np.diag(covariance) + np.r_[np.diag(pooled), np.zeros(d - p)])
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Constant or nonfinite raw features cannot identify the grouped null")
    standardized = (z - initial_mean) / scales
    covariance = covariance / scales[:, None] / scales[None, :]
    pooled = pooled / scales[:p, None] / scales[None, :p]
    scatter = scatter / scales[:p, None] / scales[None, :p]
    check_rank(covariance, config.rank_tol, "Group-mean")
    check_rank(pooled, config.rank_tol, "Pooled within-group")
    corrected = covariance.copy()
    corrected[:p, :p] -= pooled * np.mean(1 / counts)
    # Moment correction is only an initialization; the accepted fit maximizes ML.
    if np.linalg.eigvalsh(corrected)[0] <= config.rank_tol:
        corrected = covariance.copy()
    starts = [(corrected, pooled), (0.5 * covariance, 0.5 * pooled), (1.5 * covariance, 2 * pooled)]
    bounds = []
    for width in (d, p):
        bounds.extend(
            (-20, 20) if i == j else (None, None)
            for i, j in zip(*np.tril_indices(width), strict=True)
        )
    attempts, candidates = [], []
    for index, (b, r) in enumerate(starts):
        start = np.r_[pack_covariance(b), pack_covariance(r)]

        def objective(parameters):
            value, gradient, _ = profile_likelihood(parameters, standardized, counts, scatter)
            return value / g, gradient / g

        try:
            result = minimize(
                objective,
                start,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": config.max_iterations,
                    "ftol": 1e-13,
                    "gtol": 1e-8,
                    "maxls": 40,
                },
            )
            gradient_norm = float(np.max(np.abs(result.jac)))
            split = d * (d + 1) // 2
            lb, lr = _factor(result.x[:split], d), _factor(result.x[split:], p)
            b, r = lb @ lb.T, lr @ lr.T
            check_rank(b, config.rank_tol, "Between-group")
            check_rank(r, config.rank_tol, "Within-group")
            on_bound = any(
                bound[0] is not None and min(x - bound[0], bound[1] - x) <= 1e-5
                for x, bound in zip(result.x, bounds, strict=True)
            )
            accepted = bool(
                result.success
                and np.isfinite(result.fun)
                and gradient_norm <= config.gradient_tol
                and not on_bound
            )
            attempts.append(
                {
                    "start": index,
                    "accepted": accepted,
                    "message": str(result.message),
                    "iterations": int(result.nit),
                    "objective_per_group": float(result.fun),
                    "gradient_max": gradient_norm,
                }
            )
            if accepted:
                _, _, mean = profile_likelihood(result.x, standardized, counts, scatter)
                candidates.append((float(result.fun), index, mean, b, r, result.x))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            attempts.append({"start": index, "accepted": False, "error": str(exc)})
    if not candidates:
        raise ValueError(f"Grouped Gaussian ML did not yield an interior converged fit: {attempts}")
    _, index, mean, b, r, parameters = min(candidates, key=lambda item: (item[0], item[1]))
    mean = initial_mean + scales * mean
    b = b * scales[:, None] * scales[None, :]
    r = r * scales[:p, None] * scales[None, :p]
    raw_z, counts, raw_scatter = sufficient_statistics(morphology, ecology, targets)
    raw_parameters = np.r_[pack_covariance(b), pack_covariance(r)]
    likelihood, _, _ = profile_likelihood(raw_parameters, raw_z, counts, raw_scatter, constant=True)
    return (
        mean,
        b,
        r,
        {
            "selected_start": index,
            "attempts": attempts,
            "negative_log_likelihood": float(likelihood),
            "residual_degrees_of_freedom": residual_df,
            "standardization_center": initial_mean.tolist(),
            "standardization_scale": scales.tolist(),
            "standardized_parameters": parameters.tolist(),
            "optimizer": "L-BFGS-B; analytic profiled gradient; three fixed starts",
            "diagonal_log_guard": [-20, 20],
            "guard_boundary_rejected": True,
        },
    )
