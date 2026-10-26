"""Guarded fitted-tree diagnostics, distinct from production clustering.

The frontier/integral calculations audit a retained library hierarchy, not a
separate implementation of the complete HDBSCAN algorithm. See the official
scikit-learn HDBSCAN tree source linked in docs/noisy-moons-diagnostics.md.
"""

from __future__ import annotations

import math
import re
from importlib.metadata import version

import numpy as np

LINKAGE_FIELDS = ("left_node", "right_node", "value", "cluster_size")
CONDENSED_FIELDS = ("parent", "child", "value", "cluster_size")


def validate_linkage(linkage, n):
    if linkage.dtype.names != LINKAGE_FIELDS or linkage.shape != (n - 1,):
        raise ValueError("Unsupported fitted linkage representation")
    if not np.isfinite(linkage["value"]).all() or np.any(linkage["value"] <= 0):
        raise ValueError(
            "Finite strictly positive hierarchy distances required for this diagnostic"
        )
    active = {i: {i} for i in range(n)}
    members = dict(active)
    for i, row in enumerate(linkage):
        left, right = int(row["left_node"]), int(row["right_node"])
        if left == right or left not in active or right not in active:
            raise ValueError("Invalid linkage topology")
        union = active.pop(left) | active.pop(right)
        if len(union) != row["cluster_size"]:
            raise ValueError("Linkage size accounting mismatch")
        active[n + i] = union
        members[n + i] = union
    if len(active) != 1 or len(next(iter(active.values()))) != n:
        raise ValueError("Linkage does not cover all source units")
    return members


def retained_hierarchy(estimator, n, min_cluster_size):
    match = re.match(r"^1\.(\d+)\.", version("scikit-learn"))
    if not match or int(match[1]) < 6:
        raise ValueError("Unsupported scikit-learn version for fitted-tree diagnostic")
    from sklearn.cluster._hdbscan._tree import _condense_tree

    linkage = getattr(estimator, "_single_linkage_tree_", None)
    if linkage is None:
        raise ValueError("Fitted single-linkage hierarchy is unavailable")
    validate_linkage(linkage, n)
    condensed = _condense_tree(linkage, min_cluster_size)
    if condensed.dtype.names != CONDENSED_FIELDS or not len(condensed):
        raise ValueError("Unsupported condensed-tree representation")
    if not np.isfinite(condensed["value"]).all() or np.any(condensed["value"] <= 0):
        raise ValueError("Unsupported nonfinite/zero condensed-tree lifetime")
    return linkage.copy(), condensed.copy()


def selection_trace(condensed, *, method, allow_single_cluster):
    """Integrate retained branch mass and solve an ancestor-exclusive frontier."""
    root = int(condensed["parent"].min())
    nodes = sorted(set(int(x) for x in condensed["parent"]))
    birth = {root: 0.0}
    children = {x: [] for x in nodes}
    parent = {}
    departures = {}
    lifetimes = {}
    for row in condensed:
        p, c, lam, size = (
            int(row["parent"]),
            int(row["child"]),
            float(row["value"]),
            int(row["cluster_size"]),
        )
        parent[c] = p
        departures[c] = lam
        lifetimes[p] = max(lifetimes.get(p, 0), lam)
        if size > 1:
            birth[c] = lam
            children[p].append(c)
    own = {
        p: math.fsum(
            (float(row["value"]) - birth[p]) * int(row["cluster_size"])
            for row in condensed
            if int(row["parent"]) == p
        )
        for p in nodes
    }
    effective = {}
    frontiers = {}
    traces = []
    for p in reversed(nodes):
        child_score = math.fsum(effective[c] for c in children[p])
        eligible = p != root or allow_single_cluster
        if method == "eom":
            choose = eligible and own[p] >= child_score
            selected = [p] if choose else [k for c in children[p] for k in frontiers[c]]
        elif method == "leaf":
            # A hierarchy with no surviving cluster edges has no leaf frontier.
            choose = not children[p] and p != root
            selected = [p] if choose else [k for c in children[p] for k in frontiers[c]]
        else:
            raise ValueError("Unsupported diagnostic selection method")
        effective[p] = max(own[p], child_score) if eligible else child_score
        frontiers[p] = selected
        traces.append(
            {
                "node": p,
                "parent": parent.get(p),
                "birth": birth[p],
                "own_stability": own[p],
                "best_child_sum": child_score,
                "margin": own[p] - child_score,
                "locally_selected": choose,
                "children": children[p],
            }
        )
    chosen = sorted(frontiers[root])
    label_map = {c: i for i, c in enumerate(chosen)}
    labels = np.full(root, -1, dtype=int)
    strength = np.zeros(root)
    for i in range(root):
        c = parent[i]
        while c not in label_map and c != root:
            c = parent[c]
        if c not in label_map:
            continue
        if c == root and len(chosen) == 1 and departures[i] < lifetimes[root]:
            continue
        labels[i] = label_map[c]
        strength[i] = min(departures[i], lifetimes[c]) / lifetimes[c]
    return (
        {
            "root": root,
            "selected_nodes": chosen,
            "root_selected": root in chosen,
            "root_stability": own[root],
            "root_best_child_sum": math.fsum(effective[c] for c in children[root]),
            "root_margin": own[root] - math.fsum(effective[c] for c in children[root]),
            "surviving_cluster_nodes": len(nodes),
            "surviving_leaves": sum(not children[p] for p in nodes if p != root),
            "nodes": traces,
        },
        labels,
        strength,
    )
