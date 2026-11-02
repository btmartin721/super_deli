"""Version-one graph numerics; no generating labels or performance-based repair."""

import hashlib
import inspect
import time

import numpy as np
from scipy.linalg import eigh, qr, svd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.spatial.distance import cdist
from sklearn.manifold import spectral_embedding
from threadpoolctl import threadpool_limits

from super_deli.models.mixture import FIT_LOCK

GAP_TOL = 1e-8


def qr_helper():
    from sklearn.cluster._spectral import cluster_qr

    if tuple(inspect.signature(cluster_qr).parameters) != ("vectors",):
        raise ValueError("Unsupported spectral QR helper signature")
    return cluster_qr


def helper_fingerprint():
    return hashlib.sha256(inspect.getsource(qr_helper()).encode()).hexdigest()


def rotation(maps):
    _, _, pivots = qr(maps.T, pivoting=True)
    u, singular, vh = svd(maps[pivots[: maps.shape[1]], :].T)
    return np.abs(maps @ (u @ vh)), pivots, singular


def graph(x, unit_ids, config):
    if x.ndim != 2 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("Complete finite coordinates with at least two columns required")
    n, p = x.shape
    if (
        len(unit_ids) != n
        or len(set(unit_ids)) != n
        or any(not isinstance(s, str) or not s or s != s.strip() for s in unit_ids)
    ):
        raise ValueError("One unique source identity per row required")
    if (
        n < 32
        or n <= config.n_neighbors
        or not 1 < config.max_k < n
        or 96 * n * n + 32 * n * p > config.max_working_bytes
    ):
        raise ValueError("Unsupported unit count, K search or graph workspace")
    distance = cdist(x, x)
    np.fill_diagonal(distance, np.inf)
    by_id = {s: i for i, s in enumerate(sorted(unit_ids))}
    rank = np.array([by_id[s] for s in unit_ids])
    neighbors = np.array(
        [np.lexsort((rank, row))[: config.n_neighbors] for row in distance], dtype=np.int32
    )
    boundary = distance[np.arange(n), neighbors[:, -1]]
    ties = np.sum(distance == boundary[:, None], axis=1)
    b = csr_matrix(
        (np.ones(neighbors.size), (np.repeat(np.arange(n), config.n_neighbors), neighbors.ravel())),
        shape=(n, n),
    )
    affinity = ((b + b.T) * 0.5).tocsr()
    affinity.sort_indices()
    count, components = connected_components(affinity, directed=False)
    values, vectors = eigh(laplacian(affinity, normed=True).toarray(), check_finite=True)
    arrays = {
        "neighbors": neighbors,
        "indptr": affinity.indptr,
        "indices": affinity.indices,
        "data": affinity.data,
        "degrees": np.asarray(affinity.sum(axis=1)).ravel(),
        "components": components,
        "eigenvalues": values,
        "dense_eigenvectors": vectors[:, : config.max_k + 1],
    }
    return (
        affinity,
        arrays,
        {
            "n": n,
            "dimensions": p,
            "components": int(count),
            "exact_cutoff_ties": ties.tolist(),
            "cutoff_tie_rows": int(sum(ties > 1)),
        },
    )


def propose(values, max_k):
    gaps = np.diff(values[: max_k + 1])
    order = np.argsort(-gaps, kind="stable")
    k = int(order[0]) + 1
    lead = float(gaps[order[0]] - gaps[order[1]])
    reason = None
    if gaps[order[0]] <= GAP_TOL or lead <= GAP_TOL:
        reason = "unresolved_gap_choice"
    elif k == max_k:
        reason = "search_boundary"
    return gaps, {"proposed_k": k, "gap_lead": lead, "reason": reason}


def embedding(affinity, arrays, target, seed):
    maps = spectral_embedding(
        affinity,
        n_components=target,
        eigen_solver="arpack",
        random_state=seed,
        eigen_tol=0.0,
        norm_laplacian=True,
        drop_first=False,
    )
    arrays["embedding"] = maps
    if not np.isfinite(maps).all():
        raise ValueError("Nonfinite spectral embedding")
    labels = qr_helper()(maps)
    arrays["proposed_labels"] = labels
    if labels.shape != (len(maps),) or labels.dtype.kind not in "iu" or np.any(labels < 0):
        raise ValueError("Invalid QR memberships")
    rotated, pivots, singular = rotation(maps)
    ordered = np.sort(rotated, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    v = np.sqrt(arrays["degrees"])[:, None] * maps
    dense = laplacian(affinity, normed=True).toarray()
    basis = arrays["dense_eigenvectors"][:, :target]
    diagnostics = {
        "residual": float(np.linalg.norm(dense @ v - v * arrays["eigenvalues"][:target])),
        "gram_error": float(np.linalg.norm(v.T @ v - np.eye(target))),
        "projector_error": float(np.linalg.norm(v @ v.T - basis @ basis.T)),
    }
    reason = None
    if diagnostics["residual"] > 1e-8 or diagnostics["gram_error"] > 1e-8:
        reason = "unresolved_numerical_embedding"
    elif diagnostics["projector_error"] > 1e-6:
        reason = "unresolved_numerical_embedding"
    elif np.any(margin <= 1e-10 * np.maximum(1, ordered[:, -1])):
        reason = "ambiguous_qr_assignment"
    elif len(set(labels)) != target:
        reason = "requested_k_not_obtained"
    return {
        "embedding": maps,
        "proposed_labels": labels,
        "rotated": rotated,
        "pivots": pivots,
        "qr_singular_values": singular,
        "assignment_margin": margin,
    }, {**diagnostics, "reason": reason}


def fit_scores(scores, unit_ids, config):
    start = time.monotonic()
    arrays = {"scores": scores.copy()}
    result = {"status": "withheld", "selected_k": None, "solver_called": False}
    with FIT_LOCK, threadpool_limits(limits=1):
        affinity, graph_arrays, info = graph(scores, unit_ids, config)
        arrays.update(graph_arrays)
        result.update(info)
        gaps, selection = propose(arrays["eigenvalues"], config.max_k)
        arrays["gaps"] = gaps
        result.update(selection)
        k = result["proposed_k"]
        try:
            if result["reason"]:
                pass
            elif result["components"] == 1 and k == 1:
                arrays["proposed_labels"] = np.zeros(len(scores), dtype=int)
                result["method"] = "constant"
            elif result["components"] == k:
                arrays["proposed_labels"] = arrays["components"].copy()
                result["method"] = "components"
            elif result["components"] != 1:
                result["reason"] = "unsupported_component_count"
            else:
                result["solver_called"] = True
                evidence, diagnostic = embedding(affinity, arrays, k, config.random_seed)
                arrays.update(evidence)
                result.update(diagnostic)
                result["method"] = "spectral"
            if result["reason"] is None:
                labels = arrays["proposed_labels"]
                sizes = [int(sum(labels == j)) for j in sorted(set(labels))]
                result["cluster_sizes"] = sizes
                if min(sizes) < config.min_cluster_size:
                    result["reason"] = "small_cluster"
                else:
                    arrays["labels"] = labels.copy()
                    result.update(status="available", selected_k=k)
        except Exception as exc:
            result.update(status="failed", reason="solver_or_assignment_failure", error=str(exc))
    result["seconds"] = time.monotonic() - start
    return result, arrays
