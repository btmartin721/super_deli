"""Post-fit geometry and branch evaluations; references never choose a model."""

import numpy as np
from scipy.spatial.distance import cdist

from super_deli.benchmarking.moons_geometry import graph_diagnostics
from super_deli.benchmarking.moons_hierarchy import validate_linkage
from super_deli.benchmarking.view_signal_metrics import linear_recovery


def branch_agreement(linkage, condensed, truth):
    groups = np.unique(truth)
    if len(groups) < 2:
        return dict(reference_groups=1, best_branch_jaccard=None)
    n = len(truth)
    members = {i: {i} for i in range(n)}
    children = {}
    for row in condensed:
        children.setdefault(int(row["parent"]), []).append(int(row["child"]))
    for node in sorted(children, reverse=True):
        members[node] = set.union(*(members[c] for c in children[node]))
    root = min(children)
    branches = [node for node in children if node != root]
    references = [set(np.flatnonzero(truth == g)) for g in groups]
    best = [
        max((len(members[node] & g) / len(members[node] | g) for node in branches), default=0.0)
        for g in references
    ]
    all_nodes = validate_linkage(linkage, n)
    pure_sizes = [
        max((len(ids) for ids in all_nodes.values() if ids <= g), default=0) for g in references
    ]
    return dict(
        reference_groups=len(groups),
        best_branch_jaccard=best,
        largest_pure_linkage_fraction=[
            size / len(g) for size, g in zip(pure_sizes, references, strict=True)
        ],
        interpretation=(
            "Post-fit reference agreement; no truth-selected partition is fitted or selected"
        ),
    )


def diagnostics(scores, reference, linkage, condensed):
    n = len(scores)
    distance = cdist(scores, scores)
    order = np.argsort(distance, axis=1, kind="stable")
    neighbors = np.array([[j for j in row if j != i][:15] for i, row in enumerate(order)])
    saved = dict(distances=distance, neighbors_15=neighbors)
    result = {
        "linear_morphology": linear_recovery(scores, reference["latent_morphology"]),
        "linear_ecology": linear_recovery(scores, reference["latent_ecology"]),
        "effective_rank": int(
            np.linalg.matrix_rank(scores, tol=np.linalg.svd(scores, compute_uv=False)[0] * 1e-12)
        ),
        "references": {},
    }
    partitions = (
        {"common": reference["truth"]}
        if "truth" in reference
        else {role: reference[role + "_truth"] for role in ("morphology", "ecology")}
    )
    for name, truth in partitions.items():
        graph, core, edges = graph_diagnostics(scores, truth, 5)
        graph["neighbor_impurity"] = {
            str(k): float(np.mean(truth[neighbors[:, :k]] != truth[:, None]))
            if len(set(truth)) > 1
            else None
            for k in (5, 15)
        }
        result["references"][name] = dict(
            graph=graph, branches=branch_agreement(linkage, condensed, truth)
        )
        if name == next(iter(partitions)):
            saved.update(core_distances=core, mst_edges=edges)
    upper = np.triu_indices(n, 1)
    result["latent_distance_correlations"] = {}
    for role in ("morphology", "ecology"):
        baseline = cdist(reference["latent_" + role], reference["latent_" + role])
        result["latent_distance_correlations"][role] = float(
            np.corrcoef(distance[upper], baseline[upper])[0, 1]
        )
    return result, saved
