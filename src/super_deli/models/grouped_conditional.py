"""Conditional Gaussian ML with a positive-semidefinite morphology component.

Ecology has an unconstrained positive-definite Gaussian covariance. Given
observed ecology, morphology group means have a multivariate regression mean
and covariance T + R/n_g. T may lie on the PSD cone boundary; R remains PD.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from super_deli.models.grouped_gaussian import (
    _factor,
    check_rank,
    pack_covariance,
    sufficient_statistics,
)

ESTIMATOR = "conditional_psd_v1"


def project_psd(matrix):
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    return (vectors * np.maximum(values, 0)) @ vectors.T


def psd_root(matrix):
    """Symmetric root; only machine-roundoff eigenvalues are treated as zero."""
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    tolerance = 64 * np.finfo(float).eps * len(values) * max(np.max(np.abs(values)), 1.0)
    if values[0] < -tolerance:
        raise ValueError("Covariance is outside the PSD cone beyond roundoff")
    values[np.abs(values) <= tolerance] = 0
    return (vectors * np.sqrt(values)) @ vectors.T


def conditional_likelihood(between, within, y, design, counts, scatter, *, constant=False):
    """Profile regression by GLS; return NLL and covariance-matrix gradients."""
    g, p = y.shape
    k = design.shape[1]
    precision, weighted = np.zeros((k * p, k * p)), np.zeros((k, p))
    cache = []
    for n in np.unique(counts):
        selected = counts == n
        x, rows = design[selected], y[selected]
        covariance = between + within / n
        np.linalg.cholesky(covariance)
        inverse = np.linalg.inv(covariance)
        determinant = np.linalg.slogdet(covariance)[1]
        precision += np.kron(x.T @ x, inverse)
        weighted += x.T @ rows @ inverse
        cache.append((n, x, rows, inverse, determinant))
    beta = np.linalg.solve(precision, weighted.ravel()).reshape(k, p)
    value, gt, gr = 0.0, np.zeros((p, p)), np.zeros((p, p))
    for n, x, rows, inverse, determinant in cache:
        residual = rows - x @ beta
        projected = residual @ inverse
        value += 0.5 * (len(rows) * determinant + np.sum(projected * residual))
        gradient = 0.5 * (len(rows) * inverse - projected.T @ projected)
        gt += gradient
        gr += gradient / n
    np.linalg.cholesky(within)
    inverse = np.linalg.inv(within)
    df = int(counts.sum()) - g
    value += 0.5 * (df * np.linalg.slogdet(within)[1] + np.trace(inverse @ scatter))
    gr += 0.5 * (df * inverse - inverse @ scatter @ inverse)
    if constant:
        value += 0.5 * (p * np.log(counts).sum() + counts.sum() * p * np.log(2 * np.pi))
    return float(value), (gt + gt.T) / 2, (gr + gr.T) / 2, beta


def optimality(between, gt, gr):
    """Projected cone stationarity plus interior residual-covariance score."""
    cone = float(np.max(np.abs(between - project_psd(between - gt))))
    residual = float(np.max(np.abs(gr)))
    return {"cone_residual": cone, "within_score": residual, "maximum": max(cone, residual)}


def _polish(between, within, objective, config):
    value, gt, gr, beta = objective(between, within)
    step, backtracks = 1.0, 0
    termination = "iteration_limit"
    for iteration in range(config.solver.max_polish_iterations + 1):
        score = optimality(between, gt, gr)
        if score["maximum"] <= config.gradient_tol:
            termination = "constrained_stationarity"
            break
        if iteration == config.solver.max_polish_iterations:
            break
        accepted = False
        for _ in range(config.solver.max_line_search):
            trial_t = project_psd(between - step * gt)
            trial_r = project_psd(within - step * gr)
            distance = np.sum((trial_t - between) ** 2) + np.sum((trial_r - within) ** 2)
            try:
                new_value, new_gt, new_gr, new_beta = objective(trial_t, trial_r)
                if np.isfinite(new_value) and new_value <= value - 1e-4 * distance / step:
                    accepted = True
                    break
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                pass
            step *= 0.5
            backtracks += 1
        if not accepted:
            termination = "line_search_failed"
            break
        delta_t, delta_r = trial_t - between, trial_r - within
        curvature = np.sum(delta_t * (new_gt - gt)) + np.sum(delta_r * (new_gr - gr))
        step = float(np.clip(distance / curvature, 1e-8, 1e6)) if curvature > 0 else 1.0
        between, within, value, gt, gr, beta = trial_t, trial_r, new_value, new_gt, new_gr, new_beta
    return (
        between,
        within,
        value,
        beta,
        {
            "termination": termination,
            "iterations": iteration,
            "backtracks": backtracks,
            **optimality(between, gt, gr),
        },
    )


def fit_gaussian(morphology, ecology, targets, config):
    z, counts, scatter = sufficient_statistics(morphology, ecology, targets)
    g, d = z.shape
    p, q = morphology.shape[1], ecology.shape[1]
    df = len(morphology) - g
    if g <= d or g < 2 * config.model.min_cluster_size or df < p:
        raise ValueError(
            "Require G>P+Q, two feasible components and residual degrees of freedom >=P"
        )
    coefficients = p * (q + 1)
    estimate = 8 * (8 * coefficients**2 + 32 * g * d + 48 * d**2)
    if estimate > config.model.max_working_bytes:
        raise ValueError("Conditional GLS workspace exceeds max_working_bytes")
    center = z.mean(axis=0)
    raw_cov = (z - center).T @ (z - center) / g
    pooled = scatter / df
    scales = np.sqrt(np.diag(raw_cov) + np.r_[np.diag(pooled), np.zeros(q)])
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Constant or nonfinite raw features cannot identify the grouped null")
    values = (z - center) / scales
    eco_cov = values[:, p:].T @ values[:, p:] / g
    pooled = pooled / scales[:p, None] / scales[None, :p]
    scatter = scatter / scales[:p, None] / scales[None, :p]
    check_rank(eco_cov, config.rank_tol, "Ecology")
    check_rank(pooled, config.rank_tol, "Pooled within-group")
    design = np.column_stack((np.ones(g), values[:, p:]))
    y = values[:, :p]
    ols = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ ols
    raw_t = residual.T @ residual / g
    corrected = project_psd(raw_t - pooled * np.mean(1 / counts))
    starts = [
        (corrected, pooled),
        (0.5 * raw_t, 0.5 * pooled),
        (1.5 * raw_t, 2 * pooled),
        (np.zeros((p, p)), pooled),
    ]

    def objective(t, r):
        value, gt, gr, beta = conditional_likelihood(t, r, y, design, counts, scatter)
        if not np.isfinite(value) or not np.isfinite(gt).all() or not np.isfinite(gr).all():
            raise ValueError("Nonfinite conditional objective or covariance score")
        return value / g, gt / g, gr / g, beta

    attempts, candidates = [], []
    for index, (t, r) in enumerate(starts):
        attempt = {"start": index, "accepted": False}

        def factor_objective(parameters):
            factor = parameters[: p * p].reshape(p, p)
            lr = _factor(parameters[p * p :], p)
            value, gt, gr, _ = objective(factor @ factor.T, lr @ lr.T)
            grad_r = 2 * gr @ lr
            grad_r[np.diag_indices(p)] *= np.diag(lr)
            return value, np.r_[(2 * gt @ factor).ravel(), grad_r[np.tril_indices(p)]]

        try:
            initial = np.r_[psd_root(t).ravel(), pack_covariance(r)]
            bounds = [(None, None)] * (p * p) + [
                (-20, 20) if i == j else (None, None)
                for i, j in zip(*np.tril_indices(p), strict=True)
            ]
            result = minimize(
                factor_objective,
                initial,
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
            attempt["factor_stage"] = {
                "success": bool(result.success),
                "iterations": int(result.nit),
                "message": str(result.message),
            }
            factor = result.x[: p * p].reshape(p, p)
            lr = _factor(result.x[p * p :], p)
            t, r, value, beta, diagnostics = _polish(
                factor @ factor.T, lr @ lr.T, objective, config
            )
            # Canonicalize numerical zero directions, then recheck the actual stored fit.
            root = psd_root(t)
            t = root @ root.T
            value, gt, gr, beta = objective(t, r)
            score = optimality(t, gt, gr)
            check_rank(r, config.rank_tol, "Within-group")
            accepted = bool(np.isfinite(value) and score["maximum"] <= config.gradient_tol)
            attempt.update(
                accepted=accepted,
                objective_per_group=value,
                polish_stage=diagnostics,
                optimality=score,
            )
            if accepted:
                candidates.append((value, index, t, r, beta))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            attempt["error"] = str(exc)
        attempts.append(attempt)
    if not candidates:
        raise ValueError(f"Grouped conditional PSD ML has no constrained converged fit: {attempts}")
    _, index, t, r, beta = min(candidates, key=lambda item: (item[0], item[1]))
    value, gt, gr, beta = objective(t, r)
    raw_t = t * scales[:p, None] * scales[None, :p]
    raw_r = r * scales[:p, None] * scales[None, :p]
    raw_c = eco_cov * scales[p:, None] * scales[None, p:]
    regression = beta[1:].T * scales[:p, None] / scales[None, p:]
    mean = np.r_[center[:p] + scales[:p] * beta[0], center[p:]]
    factor = np.zeros((d, d))
    factor[:p, :p] = scales[:p, None] * psd_root(t)
    factor[p:, p:] = scales[p:, None] * np.linalg.cholesky(eco_cov)
    factor[:p, p:] = regression @ factor[p:, p:]
    between = factor @ factor.T
    conditional_nll = conditional_likelihood(t, r, y, design, counts, scatter, constant=True)[0]
    ecology_nll = 0.5 * g * (q * np.log(2 * np.pi) + np.linalg.slogdet(eco_cov)[1] + q)
    nll = (
        conditional_nll
        + ecology_nll
        + len(morphology) * np.log(scales[:p]).sum()
        + g * np.log(scales[p:]).sum()
    )
    eigenvalues = np.linalg.eigvalsh(t)
    rank = int(np.sum(eigenvalues > config.rank_tol * max(eigenvalues[-1], 1)))
    return (
        mean,
        between,
        raw_r,
        {
            "estimator": ESTIMATOR,
            "selected_start": index,
            "attempts": attempts,
            "negative_log_likelihood": float(nll),
            "residual_degrees_of_freedom": df,
            "standardization_center": center.tolist(),
            "standardization_scale": scales.tolist(),
            "conditional_covariance": raw_t.tolist(),
            "ecology_covariance": raw_c.tolist(),
            "regression": regression.tolist(),
            "sampling_factor": factor.tolist(),
            "within_sampling_factor": (scales[:p, None] * np.linalg.cholesky(r)).tolist(),
            "conditional_rank": rank,
            "conditional_eigenvalues": eigenvalues.tolist(),
            "boundary_or_near_boundary": rank < p,
            "optimality": optimality(t, gt, gr),
            "optimizer": (
                "Four fixed factor starts, followed by bounded projected-covariance continuation"
            ),
            "roundoff_zero_rule": (
                "64 * machine_epsilon * dimension * max(spectral_norm,1), "
                "in standardized coordinates; no positive variance floor"
            ),
            "sampler": "joint_conditional_factor_v2",
        },
    )
