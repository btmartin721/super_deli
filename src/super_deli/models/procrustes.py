"""Unit-centroid-size generalized Procrustes alignment of equally weighted fixed points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from super_deli.models.shape_config import ShapeConfig


class GPAConvergenceError(ValueError):
    def __init__(self, history: tuple[dict, ...]):
        super().__init__(f"GPA did not converge after {len(history)} iterations")
        self.history = history


@dataclass(frozen=True)
class GPAResult:
    aligned: np.ndarray
    consensus: np.ndarray
    centroids: np.ndarray
    centroid_sizes: np.ndarray
    rotations: np.ndarray
    history: tuple[dict, ...]


def coordinate_array(coordinates, config: ShapeConfig) -> np.ndarray:
    raw = np.asarray(coordinates)
    if raw.ndim != 3 or raw.shape[0] < 1 or raw.shape[1] < 3 or raw.shape[2] not in (2, 3):
        raise ValueError("Coordinates require shape (observations, >=3 landmarks, 2 or 3 axes)")
    if raw.dtype.kind not in "fiu":
        raise ValueError(
            "Coordinates must be real numbers without missing values or Boolean/string casts"
        )
    n, k, d = raw.shape
    p, q = k * d, min(n, k * d)
    estimate = 8 * (20 * n * p + 6 * q * q + 6 * p)
    if estimate > config.max_working_bytes:
        raise ValueError(f"Estimated numerical workspace {estimate} exceeds max_working_bytes")
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(
            "Coordinates must be finite; missing points need an explicit workflow policy"
        )
    return result


def preshapes(coordinates, config: ShapeConfig):
    raw = coordinate_array(coordinates, config)
    amplitude = np.max(np.abs(raw), axis=(1, 2))
    if np.any(amplitude == 0):
        raise ValueError("Zero centroid size: all landmarks coincide")
    # Normalize before summing/squaring to avoid overflow and underflow when
    # finite source coordinates have very large or very small magnitudes.
    scaled = raw / amplitude[:, None, None]
    centers = scaled.mean(axis=1)
    centered = scaled - centers[:, None, :]
    norms = np.linalg.norm(centered, axis=(1, 2))
    if np.any(norms == 0):
        raise ValueError("Zero centroid size: all landmarks coincide")
    unit = centered / norms[:, None, None]
    singular = np.linalg.svd(unit, compute_uv=False)
    ranks = (singular > config.rank_tol * singular[:, :1]).sum(axis=1)
    if np.any(ranks < raw.shape[2] - 1):
        raise ValueError("Geometric rank cannot identify rotations (3D collinear configuration)")
    with np.errstate(over="raise", invalid="raise"):
        try:
            sizes = amplitude * norms
            centroids = amplitude[:, None] * centers
        except FloatingPointError as exc:
            raise ValueError("Centroid size/position exceeds finite floating-point range") from exc
    if np.any(sizes <= 0) or not np.isfinite(sizes).all():
        raise ValueError("Centroid size is not representable as a positive finite number")
    return unit, centroids, sizes


def align_preshapes(unit: np.ndarray, reference: np.ndarray, config: ShapeConfig):
    cross = np.einsum("nki,kj->nij", unit, reference)
    u, s, vh = np.linalg.svd(cross)
    ranks = (s > config.rank_tol * s[:, :1]).sum(axis=1)
    if np.any(ranks < unit.shape[2] - 1):
        raise ValueError("Cross-covariance rank cannot identify a unique alignment rotation")
    if config.reflection == "forbid":
        signs = np.where(np.linalg.det(u @ vh) < 0, -1.0, 1.0)
        ambiguous = (signs < 0) & ((s[:, -2] - s[:, -1]) <= config.rank_tol * s[:, 0])
        if np.any(ambiguous):
            raise ValueError("Proper rotation is nonunique under the reflection restriction")
        u[:, :, -1] *= signs[:, None]
    rotations = u @ vh
    return unit @ rotations, rotations


def _unit_mean(aligned: np.ndarray) -> np.ndarray:
    mean = aligned.mean(axis=0)
    mean -= mean.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm <= np.finfo(float).eps:
        raise ValueError("Degenerate consensus; check landmark correspondence and shape dispersion")
    return mean / norm


def generalized_procrustes(coordinates, *, config: ShapeConfig | None = None) -> GPAResult:
    """Remove translation/size and iteratively align to the normalized mean.

    The first observation anchors the otherwise arbitrary global orientation.
    All observations/landmarks have equal weight; optional reflections are explicit.
    Convergence is numerical stationarity, not a guarantee of a global optimum.
    """
    config = config or ShapeConfig()
    unit, centroids, sizes = preshapes(coordinates, config)
    reference, history = unit[0].copy(), []
    for iteration in range(1, config.max_iter + 1):
        aligned, _ = align_preshapes(unit, reference, config)
        updated = _unit_mean(aligned)
        delta = float(np.linalg.norm(updated - reference))
        history.append(
            {
                "iteration": iteration,
                "consensus_delta": delta,
                "sum_squared_residuals": float(np.sum((aligned - updated) ** 2)),
            }
        )
        reference = updated
        if delta <= config.tol:
            aligned, rotations = align_preshapes(unit, reference, config)
            if np.linalg.norm(_unit_mean(aligned) - reference) <= config.tol:
                return GPAResult(aligned, reference, centroids, sizes, rotations, tuple(history))
    raise GPAConvergenceError(tuple(history))
