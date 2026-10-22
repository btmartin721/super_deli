"""Shared BIC selection for Gaussian mixtures on a fixed feature space."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from threading import RLock
from typing import Protocol

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

# threadpoolctl changes process-wide settings; serialize our numerical fits.
FIT_LOCK = RLock()


class MixtureSettings(Protocol):
    max_k: int
    min_cluster_size: int
    n_init: int
    max_iter: int
    tol: float
    reg_covar: float


@dataclass(frozen=True)
class MixtureFit:
    labels: np.ndarray
    membership: np.ndarray
    candidates: tuple[dict, ...]
    mixture: GaussianMixture
    component_order: tuple[int, ...]

    @property
    def selected_k(self) -> int:
        return self.membership.shape[1]


def select_gaussian_mixture(
    scores: np.ndarray,
    config: MixtureSettings,
    seed: int,
    *,
    covariance_type: str,
    unit_indices: np.ndarray | None = None,
) -> MixtureFit:
    """Include K=1; fail the search if any feasible candidate fails numerically.

    Bootstrap multiplicities enter the likelihood, but minimum component size
    counts distinct source units when ``unit_indices`` is supplied.
    """
    candidates, accepted = [], []
    unique_count = len(scores) if unit_indices is None else len(np.unique(unit_indices))
    for k in range(1, config.max_k + 1):
        if unique_count < k * config.min_cluster_size:
            candidates.append({"k": k, "status": "insufficient_samples", "bic": None})
            continue
        estimator = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            n_init=config.n_init,
            max_iter=config.max_iter,
            tol=config.tol,
            reg_covar=config.reg_covar,
            random_state=int(np.random.SeedSequence([seed, k]).generate_state(1)[0]),
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                estimator.fit(scores)
            if not estimator.converged_:
                raise ValueError(f"GMM K={k} did not converge")
            probabilities = estimator.predict_proba(scores)
            bic = float(estimator.bic(scores))
            if not np.isfinite(bic) or not np.isfinite(probabilities).all():
                raise ValueError(f"GMM K={k} produced nonfinite estimates")
        except (ValueError, ConvergenceWarning, np.linalg.LinAlgError, FloatingPointError) as exc:
            raise ValueError(f"Candidate K={k} failed: {exc}") from exc
        labels = probabilities.argmax(axis=1)
        sizes = np.bincount(labels, minlength=k)
        distinct = (
            sizes
            if unit_indices is None
            else np.array([len(np.unique(unit_indices[labels == c])) for c in range(k)])
        )
        status = "eligible" if distinct.min() >= config.min_cluster_size else "small_component"
        candidate = {
            "k": k,
            "status": status,
            "bic": bic,
            "converged": True,
            "iterations": int(estimator.n_iter_),
            "hard_cluster_sizes": sizes.tolist(),
        }
        if unit_indices is not None:
            candidate["distinct_unit_sizes"] = distinct.tolist()
        candidates.append(candidate)
        if status == "eligible":
            accepted.append((bic, k, estimator, probabilities))
    if not accepted:
        raise ValueError("No admissible mixture; increase sample size or reduce constraints")
    _, _, mixture, membership = min(accepted, key=lambda item: (item[0], item[1]))
    labels = membership.argmax(axis=1)
    order = tuple(
        sorted(range(mixture.n_components), key=lambda k: int(np.flatnonzero(labels == k)[0]))
    )
    membership = membership[:, order]
    return MixtureFit(membership.argmax(axis=1), membership, tuple(candidates), mixture, order)
