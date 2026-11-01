"""One density hierarchy per fixed space; policy replay never refits a graph."""

import time

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from super_deli.benchmarking.moons_geometry import branch_reference_diagnostics, graph_diagnostics
from super_deli.benchmarking.moons_hierarchy import retained_hierarchy, selection_trace
from super_deli.benchmarking.neural_moons_config import POLICIES
from super_deli.benchmarking.neural_moons_coordinates import neighborhood_metrics
from super_deli.models.mixture import FIT_LOCK


def recovery(labels, truth):
    selected = labels >= 0
    full = labels.copy()
    full[~selected] = np.arange(sum(~selected)) + max(int(labels.max()), 0) + 1
    return {
        "selected_k": len(set(labels) - {-1}),
        "noise_count": int(sum(~selected)),
        "coverage": float(selected.mean()),
        "ari": float(adjusted_rand_score(truth, full)),
        "assigned_ari": float(adjusted_rand_score(truth[selected], labels[selected]))
        if sum(selected) >= 2
        else None,
        "splits": int(sum(len(set(labels[(truth == t) & selected])) > 1 for t in set(truth))),
        "merges": int(sum(len(set(truth[labels == t])) > 1 for t in set(labels[selected]))),
    }


def fit_coordinates(scores, context, *, config, expected=None, replay_only=False):
    start = time.monotonic()
    scores = np.asarray(scores, dtype=float)
    if (
        scores.ndim != 2
        or len(scores) < 10
        or scores.shape[1] not in {2, 8}
        or not np.isfinite(scores).all()
    ):
        raise ValueError("Unsupported diagnostic coordinates")
    if 64 * len(scores) ** 2 + 160 * scores.size > config.max_working_bytes:
        raise ValueError("Diagnostic coordinate workspace exceeds budget")
    with FIT_LOCK, threadpool_limits(limits=1):
        estimator = HDBSCAN(
            min_cluster_size=5,
            min_samples=5,
            metric="euclidean",
            alpha=1.0,
            cluster_selection_epsilon=0.0,
            cluster_selection_method="eom",
            allow_single_cluster=True,
            n_jobs=1,
            copy=True,
        ).fit(scores)
        labels, strength = estimator.labels_.copy(), estimator.probabilities_.copy()
        if (
            np.any(labels < -1)
            or not np.isfinite(strength).all()
            or np.any((strength < 0) | (strength > 1))
            or np.any(strength[labels < 0] != 0)
        ):
            raise ValueError("Invalid diagnostic density labels/strength")
        if expected is not None:
            np.testing.assert_array_equal(labels, expected[0])
            np.testing.assert_array_equal(strength, expected[1])
        graph, core, edges = graph_diagnostics(scores, context["truth"], 5)
        data = {
            "scores": scores.copy(),
            "core_distances": core,
            "mst_edges": edges,
            "baseline_labels": labels,
            "baseline_strength": strength,
        }
        neighborhood, neighbors = neighborhood_metrics(scores, context)
        data.update(neighbors)
        record = {
            "status": "complete",
            "fit_n": len(scores),
            "dimensions": scores.shape[1],
            "graph": graph,
            "neighborhood": neighborhood,
            "archive_replay": expected is not None,
            "policies": {"baseline": {**recovery(labels, context["truth"]), "status": "complete"}},
            "parameters": {
                "min_cluster_size": 5,
                "min_samples": 5,
                "alpha": 1,
                "epsilon": 0,
                "metric": "euclidean",
                "self_in_min_samples": True,
            },
        }
        try:
            linkage, condensed = retained_hierarchy(estimator, len(scores), 5)
            data.update(linkage=linkage, condensed=condensed)
            record["branch_reference"] = branch_reference_diagnostics(condensed, context["truth"])
            for policy, (method, root) in POLICIES.items():
                if replay_only and policy != "baseline":
                    continue
                trace, assigned, strengths = selection_trace(
                    condensed, method=method, allow_single_cluster=root
                )
                if policy == "baseline":
                    np.testing.assert_array_equal(assigned, labels)
                    np.testing.assert_allclose(strengths, strength, atol=1e-12)
                data[policy + "_labels"] = assigned
                data[policy + "_strength"] = strengths
                record["policies"][policy] = {
                    **recovery(assigned, context["truth"]),
                    "status": "complete",
                    "selection": trace,
                }
            record["hierarchy_status"] = "available"
        except (ValueError, AssertionError, AttributeError, ImportError) as exc:
            record.update(hierarchy_status="unavailable", hierarchy_error=str(exc))
            for policy in POLICIES:
                if policy != "baseline" and not replay_only:
                    record["policies"][policy] = {"status": "unavailable", "error": str(exc)}
        record["seconds"] = time.monotonic() - start
        return record, data
