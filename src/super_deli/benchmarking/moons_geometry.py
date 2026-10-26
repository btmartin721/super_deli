"""Post-fit oracle geometry; generating labels never enter clustering."""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist


def graph_diagnostics(scores, truth, min_samples):
    d = cdist(scores, scores)
    core = np.sort(d, axis=1)[:, min_samples - 1]
    mutual = np.maximum(np.maximum(d, core[:, None]), core[None, :])
    np.fill_diagonal(mutual, 0)
    result = {
        "core_min": float(core.min()),
        "core_median": float(np.median(core)),
        "core_max": float(core.max()),
        "self_included_in_min_samples": True,
    }
    tree = minimum_spanning_tree(mutual).tocoo()
    edges = np.column_stack([tree.row, tree.col, tree.data])
    if truth is not None and len(np.unique(truth)) > 1:
        between = truth[:, None] != truth[None, :]
        nearest = np.argsort(d, axis=1)[:, 1:min_samples]
        result["cross_neighbor_fraction"] = float(np.mean(truth[nearest] != truth[:, None]))
        for key, matrix in [("euclidean", d), ("mutual", mutual)]:
            bridge = float(matrix[between].min())
            within = []
            for k in np.unique(truth):
                ids = np.flatnonzero(truth == k)
                values = minimum_spanning_tree(matrix[np.ix_(ids, ids)]).data
                within.append(float(values.max()) if len(values) else 0.0)
            bottleneck = max(within)
            result[key + "_between_bridge"] = bridge
            result[key + "_within_bottleneck"] = bottleneck
            result[key + "_separation_ratio"] = bridge / bottleneck if bottleneck else None
            result[key + "_threshold_separable"] = bridge > bottleneck
    else:
        result["cross_neighbor_fraction"] = None
    return result, core, edges


def branch_reference_diagnostics(condensed, truth):
    """Post-fit oracle branch agreement; never a production selection rule."""
    root = int(condensed["parent"].min())
    groups = np.unique(truth)
    if len(groups) != 2:
        return {"best_branch_jaccard": None, "best_disjoint_min_jaccard": None}
    child = {}
    for row in condensed:
        child.setdefault(int(row["parent"]), []).append(int(row["child"]))
    leaves = {i: {i} for i in range(root)}
    for node in sorted(child, reverse=True):
        leaves[node] = set.union(*(leaves[c] for c in child[node]))
    candidates = [c for c in child if c != root]
    reference = [set(np.flatnonzero(truth == k)) for k in groups]
    score = {c: [len(leaves[c] & g) / len(leaves[c] | g) for g in reference] for c in candidates}
    best = [max((score[c][j] for c in candidates), default=0.0) for j in (0, 1)]
    disjoint = max(
        (
            min(score[a][0], score[b][1])
            for a in candidates
            for b in candidates
            if not leaves[a] & leaves[b]
        ),
        default=0.0,
    )
    return {
        "best_branch_jaccard": best,
        "best_disjoint_min_jaccard": disjoint,
        "interpretation": "oracle reference score after fitting; not an admissible selection rule",
    }
