"""Baseline density fits in fixed coordinates, with post-fit geometry only."""

import time

import numpy as np
from sklearn.cluster import HDBSCAN
from threadpoolctl import threadpool_limits

from super_deli.benchmarking.moons_geometry import graph_diagnostics
from super_deli.benchmarking.moons_hierarchy import retained_hierarchy, selection_trace
from super_deli.benchmarking.neural_moons_fit import recovery
from super_deli.benchmarking.view_signal_metrics import branch_reference, geometry
from super_deli.models.density_scores import fit_density_scores
from super_deli.models.hdbscan_config import HDBSCANConfig
from super_deli.models.mixture import FIT_LOCK


def fit_coordinates(scores, context, *, expected=None):
    start = time.monotonic()
    scores = np.asarray(scores, dtype=float)
    if (
        scores.ndim != 2
        or scores.shape[0] != len(context["truth"])
        or scores.shape[1] not in (2, 4, 8, 12)
        or not np.isfinite(scores).all()
        or 64 * len(scores) ** 2 + 160 * scores.size > 256 * 1024**2
    ):
        raise ValueError("Invalid/oversized fixed coordinates")
    captured = {}

    def estimator(**kwargs):
        captured["fit"] = HDBSCAN(**kwargs)
        return captured["fit"]

    with FIT_LOCK, threadpool_limits(limits=1):
        labels, strength = fit_density_scores(
            scores, HDBSCANConfig(True), estimator_class=estimator
        )
        if expected is not None:
            np.testing.assert_array_equal(labels, expected[0])
            np.testing.assert_array_equal(strength, expected[1])
        # Evaluation labels first enter after the unsupervised decision.
        graph, core, edges = graph_diagnostics(scores, context["truth"], 5)
        metrics, neighbors = geometry(scores, context)
        data = {
            "scores": scores.copy(),
            "labels": labels,
            "strength": strength,
            "core_distances": core,
            "mst_edges": edges,
            **neighbors,
        }
        record = {
            "status": "complete",
            "native_replay": expected is not None,
            "dimensions": scores.shape[1],
            "recovery": recovery(labels, context["truth"]),
            "graph": graph,
            "geometry": metrics,
            "parameters": {
                "min_cluster_size": 5,
                "min_samples": 5,
                "self_in_min_samples": True,
                "metric": "euclidean",
                "alpha": 1,
                "epsilon": 0,
                "method": "eom",
                "root": True,
            },
        }
        try:
            linkage, condensed = retained_hierarchy(captured["fit"], len(scores), 5)
            trace, replay, support = selection_trace(
                condensed, method="eom", allow_single_cluster=True
            )
            np.testing.assert_array_equal(replay, labels)
            np.testing.assert_allclose(support, strength, atol=1e-12)
            data.update(linkage=linkage, condensed=condensed)
            record.update(
                hierarchy_status="available",
                selection=trace,
                branch_reference=branch_reference(condensed, context["truth"]),
            )
        except (ValueError, AssertionError, AttributeError, ImportError) as exc:
            record.update(hierarchy_status="unavailable", hierarchy_error=str(exc))
        record["seconds"] = time.monotonic() - start
        return record, data
