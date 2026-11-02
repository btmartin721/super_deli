"""Supplied-K graph partitions; truth is never a fitting argument."""

from __future__ import annotations

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

from super_deli.benchmarking.spectral_config import SETTINGS, TARGETS
from super_deli.models.mixture import FIT_LOCK


def qr_helper():
    from sklearn.cluster._spectral import cluster_qr

    if tuple(inspect.signature(cluster_qr).parameters) != ("vectors",):
        raise ValueError("Unsupported spectral QR helper signature")
    return cluster_qr


def helper_fingerprint():
    return hashlib.sha256(inspect.getsource(qr_helper()).encode()).hexdigest()


def rotation(maps):
    _, _, piv = qr(maps.T, pivoting=True)
    u, s, vh = svd(maps[piv[: maps.shape[1]], :].T)
    rotated = np.abs(maps @ (u @ vh))
    return rotated, piv, s


def build_graph(scores, unit_ids, k, *, max_working_bytes):
    x = np.asarray(scores, dtype=float)
    if (
        x.ndim != 2
        or x.shape[1] not in (2, 8)
        or len(x) != len(unit_ids)
        or len(x) <= k
        or not np.isfinite(x).all()
    ):
        raise ValueError("Complete aligned 2D/8D diagnostic coordinates required")
    if any(not isinstance(u, str) or not u for u in unit_ids) or len(set(unit_ids)) != len(
        unit_ids
    ):
        raise ValueError("Unique source identities required")
    if type(k) is not int or k not in (5, 15) or 96 * len(x) ** 2 > max_working_bytes:
        raise ValueError("Graph settings/workspace exceed the diagnostic contract")
    distance = cdist(x, x)
    np.fill_diagonal(distance, np.inf)
    ranks = np.array([sorted(unit_ids).index(u) for u in unit_ids])
    neighbors = np.array([np.lexsort((ranks, row))[:k] for row in distance], dtype=np.int32)
    boundary = distance[np.arange(len(x)), neighbors[:, -1]]
    ties = np.sum(distance == boundary[:, None], axis=1)
    directed = csr_matrix(
        (np.ones(len(x) * k), (np.repeat(np.arange(len(x)), k), neighbors.ravel())),
        shape=(len(x), len(x)),
    )
    affinity = ((directed + directed.T) * 0.5).tocsr()
    affinity.sort_indices()
    components, labels = connected_components(affinity, directed=False)
    dense = laplacian(affinity, normed=True).toarray()
    values, vectors = eigh(dense, check_finite=True)
    data = {
        "neighbors": neighbors,
        "indptr": affinity.indptr,
        "indices": affinity.indices,
        "data": affinity.data,
        "degrees": np.asarray(affinity.sum(axis=1)).ravel(),
        "components": labels,
        "eigenvalues": values,
        "dense_eigenvectors": vectors[:, :5],
    }
    info = {
        "components": int(components),
        "cutoff_tie_rows": int(sum(ties > 1)),
        "exact_cutoff_ties": ties.tolist(),
        "n": len(x),
        "dimensions": x.shape[1],
    }
    return affinity, data, info


def fit_partition(affinity, data, info, target):
    start = time.monotonic()
    result = {"target_k": target, "status": "withheld", "solver_called": False}
    evidence = {}
    if type(target) is not int or target not in TARGETS:
        raise ValueError("Only supplied K=2 and K=4 are supported")
    try:
        if info["components"] == target:
            evidence["labels"] = data["components"].copy()
            result.update(status="available", method="components")
        elif info["components"] != 1:
            result["reason"] = "unsupported_component_count"
        elif (
            data["eigenvalues"][target] - data["eigenvalues"][target - 1] <= SETTINGS["cutoff_gap"]
        ):
            result["reason"] = "unresolved_cutoff_eigenspace"
        else:
            result["solver_called"] = True
            maps = spectral_embedding(
                affinity,
                n_components=target,
                eigen_solver="arpack",
                random_state=71323,
                eigen_tol=0.0,
                norm_laplacian=True,
                drop_first=False,
            )
            evidence["embedding"] = maps
            if not np.isfinite(maps).all():
                raise ValueError("Nonfinite spectral embedding")
            proposed = qr_helper()(maps)
            rotated, piv, s = rotation(maps)
            evidence.update(
                proposed_labels=proposed, rotated=rotated, pivots=piv, qr_singular_values=s
            )
            ordered = np.sort(rotated, axis=1)
            margin = ordered[:, -1] - ordered[:, -2]
            evidence["assignment_margin"] = margin
            v = np.sqrt(data["degrees"])[:, None] * maps
            dense = laplacian(affinity, normed=True).toarray()
            residual = float(np.linalg.norm(dense @ v - v * data["eigenvalues"][:target]))
            gram = float(np.linalg.norm(v.T @ v - np.eye(target)))
            projector = float(
                np.linalg.norm(
                    v @ v.T
                    - data["dense_eigenvectors"][:, :target]
                    @ data["dense_eigenvectors"][:, :target].T
                )
            )
            result.update(residual=residual, gram_error=gram, projector_error=projector)
            if residual > 1e-8 or gram > 1e-8 or projector > 1e-6:
                result["reason"] = "unresolved_numerical_embedding"
            elif np.any(margin <= 1e-10 * np.maximum(1, ordered[:, -1])):
                result["reason"] = "ambiguous_qr_assignment"
            elif len(set(proposed)) != target:
                result["reason"] = "requested_k_not_obtained"
            else:
                evidence["labels"] = proposed.copy()
                result.update(status="available", method="spectral")
        if result["status"] == "available":
            sizes = [int(sum(evidence["labels"] == j)) for j in set(evidence["labels"])]
            result.update(cluster_sizes=sizes, small_clusters=sum(n < 5 for n in sizes))
    except Exception as exc:
        result.update(status="failed", reason="solver_or_assignment_failure", error=str(exc))
    result["seconds"] = time.monotonic() - start
    return result, evidence


def fit_graph(scores, unit_ids, k, *, max_working_bytes):
    start = time.monotonic()
    with FIT_LOCK, threadpool_limits(limits=1):
        affinity, data, info = build_graph(scores, unit_ids, k, max_working_bytes=max_working_bytes)
        records = []
        for target in TARGETS:
            r, a = fit_partition(affinity, data, info, target)
            records.append(r)
            data.update({f"K{target}_{key}": value for key, value in a.items()})
    return {**info, "partitions": records, "seconds": time.monotonic() - start}, data
