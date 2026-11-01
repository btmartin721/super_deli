"""Shared density assignment on already fitted numerical representations."""

from __future__ import annotations

import numpy as np


def fit_density_scores(scores, config, *, estimator_class=None):
    if estimator_class is None:
        from sklearn.cluster import HDBSCAN

        estimator_class = HDBSCAN
    estimator = estimator_class(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        metric="euclidean",
        alpha=config.alpha,
        cluster_selection_epsilon=float(config.cluster_selection_epsilon),
        cluster_selection_method=config.cluster_selection_method,
        allow_single_cluster=config.allow_single_cluster,
        n_jobs=1,
        copy=True,
    ).fit(scores)
    labels, strength = estimator.labels_, estimator.probabilities_
    if (
        np.any(labels < -1)
        or not np.isfinite(strength).all()
        or np.any((strength < 0) | (strength > 1))
        or np.any(strength[labels < 0] != 0)
    ):
        raise ValueError("Invalid density labels or membership strength")
    return labels.copy(), strength.copy()
