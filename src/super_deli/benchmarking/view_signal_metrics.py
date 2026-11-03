"""Post-fit reference summaries; none of these metrics selects a partition."""

import itertools

import numpy as np
from scipy.spatial.distance import cdist


def neighbors(scores, k):
    distance = cdist(scores, scores)
    np.fill_diagonal(distance, np.inf)
    order = np.argsort(distance, axis=1)
    cutoff = distance[np.arange(len(scores)), order[:, k - 1]]
    ties = np.sum(np.isclose(distance, cutoff[:, None], rtol=1e-12, atol=1e-12), axis=1) > 1
    return order[:, :k], ties


def scatter_fraction(x, labels):
    centered = x - x.mean(0)
    total = np.sum(centered**2)
    between = sum(
        np.sum(labels == g) * np.sum(centered[labels == g].mean(0) ** 2) for g in np.unique(labels)
    )
    return float(between / total) if total > 0 else None


def linear_recovery(x, target):
    z, y = x - x.mean(0), target - target.mean(0)
    coefficient, _, rank, _ = np.linalg.lstsq(z, y, rcond=None)
    residual = np.sum((y - z @ coefficient) ** 2, axis=0)
    total = np.sum(y**2, axis=0)
    return {
        "rank": int(rank),
        "dimensions": x.shape[1],
        "r2": (1 - residual / total).tolist(),
        "interpretation": "in-sample linear recovery; not total nonlinear information",
    }


def geometry(scores, context):
    truth = context["truth"]
    result = {
        "scatter_joint": scatter_fraction(scores, truth),
        "scatter_axis_m": scatter_fraction(scores, truth // 2),
        "scatter_axis_e": scatter_fraction(scores, truth % 2),
        "linear_morphology": linear_recovery(scores, context["latent_morphology"]),
        "linear_ecology": linear_recovery(scores, context["latent_ecology"]),
    }
    retained = {}
    for k in (5, 15):
        index, ties = neighbors(scores, k)
        reference, ref_ties = neighbors(context["morphology_standard"], k)
        retained[f"neighbors_{k}"] = index
        retained[f"ties_{k}"] = ties
        valid = ~ties
        common = valid & ~ref_ties
        result[f"neighbors_{k}"] = {
            "available_rows": int(sum(valid)),
            "tied_rows": int(sum(ties)),
            "overlap_available_rows": int(sum(common)),
            "purity": float(np.mean(truth[index[valid]] == truth[valid, None]))
            if any(valid)
            else None,
            "morphology_overlap": float(
                np.mean(
                    [
                        len(set(a) & set(b)) / k
                        for a, b in zip(index[common], reference[common], strict=True)
                    ]
                )
            )
            if any(common)
            else None,
        }
    return result, retained


def branch_sets(condensed, n):
    children = {}
    for row in condensed:
        children.setdefault(int(row["parent"]), []).append(int(row["child"]))
    members = {i: {i} for i in range(n)}
    for node in sorted(children, reverse=True):
        members[node] = set.union(*(members[c] for c in children[node]))
    root = min(children)
    return {node: members[node] for node in children if node != root}


def branch_reference(condensed, truth):
    candidates = branch_sets(condensed, len(truth))
    refs = [set(np.flatnonzero(truth == g)) for g in range(4)]
    scores = {node: [len(ids & g) / len(ids | g) for g in refs] for node, ids in candidates.items()}
    passing = [[node for node, values in scores.items() if values[g] >= 0.8] for g in range(4)]
    # Only four reference classes. Laminarity makes this small; the Cartesian
    # guard also explicitly enforces disjointness rather than assuming it.
    disjoint = any(
        all(not (candidates[a] & candidates[b]) for a, b in itertools.combinations(nodes, 2))
        for nodes in itertools.product(*passing)
    )
    return {
        "best_jaccard": [max((v[g] for v in scores.values()), default=0.0) for g in range(4)],
        "four_disjoint_at_0_8": disjoint,
        "interpretation": "post-fit oracle branches; never a selection rule",
    }
