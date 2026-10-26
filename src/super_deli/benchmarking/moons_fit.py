"""Instrumented diagnostic adapter; production fitting modules remain untouched."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from sklearn.cluster import HDBSCAN
from threadpoolctl import threadpool_limits

from super_deli.benchmarking.moons_config import arm_settings
from super_deli.benchmarking.moons_hierarchy import retained_hierarchy, selection_trace
from super_deli.models.mixture import FIT_LOCK
from super_deli.models.tabular_gmm import _matrix, fit_tabular_pca
from super_deli.models.tabular_hdbscan import DensityPoint, point_arrays


def fit_diagnostic_arm(table, arm):
    config, scaling = arm_settings(arm)
    prep = replace(config.preprocessing_config(), scaling=scaling)
    x, names, types = _matrix(table, prep)
    if x.shape[1] != 2 or set(types) != {"continuous"} or not np.isfinite(x).all():
        raise ValueError("Diagnostic requires exactly two complete continuous features")
    if 64 * len(x) ** 2 > config.max_working_bytes:
        raise ValueError("Diagnostic graph workspace exceeds the declared budget")
    with FIT_LOCK, threadpool_limits(limits=1):
        model, scores = fit_tabular_pca(x, names, types, prep)
        params = {
            "min_cluster_size": config.min_cluster_size,
            "min_samples": config.min_samples,
            "metric": "euclidean",
            "alpha": config.alpha,
            "cluster_selection_epsilon": float(config.cluster_selection_epsilon),
            "cluster_selection_method": config.cluster_selection_method,
            "allow_single_cluster": config.allow_single_cluster,
            "n_jobs": 1,
            "copy": True,
        }
        estimator = HDBSCAN(**params).fit(scores)
        labels, strength = estimator.labels_, estimator.probabilities_
        if np.any(labels < -1) or not np.isfinite(strength).all():
            raise ValueError("Invalid density outcome")
        point = DensityPoint(model, scores, labels.copy(), strength.copy())
        arrays = point_arrays(point)
        metadata = {
            "status": "complete",
            "parameters": params,
            "scaling": scaling,
            "selected_k": point.selected_k,
            "noise_count": int(np.sum(labels < 0)),
        }
        try:
            linkage, condensed = retained_hierarchy(estimator, len(x), config.min_cluster_size)
            trace, reconstructed, reconstructed_strength = selection_trace(
                condensed,
                method=config.cluster_selection_method,
                allow_single_cluster=config.allow_single_cluster,
            )
            np.testing.assert_array_equal(reconstructed, labels)
            np.testing.assert_allclose(reconstructed_strength, strength, rtol=1e-10, atol=1e-12)
            arrays.update(linkage=linkage, condensed=condensed)
            metadata.update(hierarchy_status="available", selection=trace)
        except (ValueError, AssertionError, AttributeError, ImportError) as exc:
            metadata.update(hierarchy_status="unavailable", hierarchy_error=str(exc))
    return metadata, arrays
