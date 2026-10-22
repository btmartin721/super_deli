"""Constrained-null checks from independent dense raw-data covariance scores."""

from __future__ import annotations

import numpy as np


def verify_boundary(
    diagnostics, between, within, mean, factor, residual_factor, gb, gr, gm, ecology, g, p, policy
):
    scales = np.asarray(diagnostics["standardization_scale"])
    sm, se = scales[:p], scales[p:]
    t = np.asarray(diagnostics["conditional_covariance"])
    c = np.asarray(diagnostics["ecology_covariance"])
    a = np.asarray(diagnostics["regression"])
    standardized_t = t / sm[:, None] / sm[None, :]
    standardized_r = within / sm[:, None] / sm[None, :]
    standardized_c = c / se[:, None] / se[None, :]
    q = len(se)
    if (
        diagnostics["estimator"] != "conditional_psd_v1"
        or diagnostics["sampler"] != "joint_conditional_factor_v2"
    ):
        raise ValueError("Unknown boundary estimator/sampler version")
    np.testing.assert_allclose(mean[p:], ecology.mean(axis=0), rtol=1e-12, atol=1e-12)
    centered = (ecology - mean[p:]) / se
    np.testing.assert_allclose(standardized_c, centered.T @ centered / g, rtol=1e-11, atol=1e-11)
    for covariance, name in ((standardized_r, "within"), (standardized_c, "ecology")):
        eigenvalues = np.linalg.eigvalsh(covariance)
        if eigenvalues[0] <= policy.rank_tol * max(eigenvalues[-1], 1):
            raise ValueError(f"Nonpositive {name} covariance in boundary fit")
    # Validate the versioned sampler factor structurally and against every covariance block.
    normalized = factor / scales[:, None]
    np.testing.assert_array_equal(normalized[p:, :p], np.zeros((q, p)))
    root = normalized[:p, :p]
    np.testing.assert_allclose(root, root.T, atol=1e-11)
    if np.linalg.eigvalsh(root)[0] < -1e-11:
        raise ValueError("Conditional sampling root is not symmetric PSD")
    np.testing.assert_allclose(root @ root.T, standardized_t, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(
        normalized[p:, p:], np.linalg.cholesky(standardized_c), rtol=1e-10, atol=1e-11
    )
    np.testing.assert_allclose(
        normalized[:p, p:],
        (a / sm[:, None] * se[None, :]) @ normalized[p:, p:],
        rtol=1e-10,
        atol=1e-11,
    )
    normalized_within = residual_factor / sm[:, None]
    np.testing.assert_allclose(
        normalized_within, np.linalg.cholesky(standardized_r), rtol=1e-10, atol=1e-11
    )
    np.testing.assert_allclose(
        normalized @ normalized.T,
        between / scales[:, None] / scales[None, :],
        rtol=1e-10,
        atol=1e-11,
    )
    # Chain dense raw joint scores to the conditional (T,A,C) parameterization.
    gt = gb[:p, :p] * sm[:, None] * sm[None, :] / g
    gwr = gr * sm[:, None] * sm[None, :] / g
    ga = (2 * gb[:p, :p] @ a @ c + 2 * gb[:p, p:] @ c) * sm[:, None] / se[None, :] / g
    gc = (
        (a.T @ gb[:p, :p] @ a + a.T @ gb[:p, p:] + gb[p:, :p] @ a + gb[p:, p:])
        * se[:, None]
        * se[None, :]
        / g
    )
    values, vectors = np.linalg.eigh((standardized_t - gt + (standardized_t - gt).T) / 2)
    projection = (vectors * np.maximum(values, 0)) @ vectors.T
    expected = {
        "cone_residual": float(np.max(np.abs(standardized_t - projection))),
        "within_score": float(np.max(np.abs(gwr))),
    }
    expected["maximum"] = max(expected.values())
    score = max(
        expected["maximum"], np.max(np.abs(ga)), np.max(np.abs(gc)), np.max(np.abs(gm * scales / g))
    )
    if not np.isfinite(score) or score > policy.gradient_tol:
        raise ValueError("Dense covariance-cone/regression/ecology optimality checks fail")
    for key, value in expected.items():
        np.testing.assert_allclose(diagnostics["optimality"][key], value, atol=2e-8, rtol=1e-3)
    eigenvalues = np.linalg.eigvalsh(standardized_t)
    tolerance = 64 * np.finfo(float).eps * p * max(np.max(np.abs(eigenvalues)), 1)
    if eigenvalues[0] < -tolerance:
        raise ValueError("Conditional covariance is outside the PSD cone")
    rank = int(np.sum(eigenvalues > policy.rank_tol * max(eigenvalues[-1], 1)))
    if diagnostics["conditional_rank"] != rank or diagnostics["boundary_or_near_boundary"] != (
        rank < p
    ):
        raise ValueError("Conditional covariance rank diagnostics disagree")
    accepted = [a for a in diagnostics["attempts"] if a["accepted"]]
    selected = next(a for a in accepted if a["start"] == diagnostics["selected_start"])
    if selected != min(accepted, key=lambda a: (a["objective_per_group"], a["start"])):
        raise ValueError("Selected constrained optimum is not the best accepted start")
    for key, value in expected.items():
        np.testing.assert_allclose(selected["optimality"][key], value, atol=2e-8, rtol=1e-3)
