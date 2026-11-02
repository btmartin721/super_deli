"""Independent graph arithmetic and selection/QR policy checks."""

import numpy as np

from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.spectral_audit import component_labels


def audit_geometry(x, unit_ids, config, r, a):
    n = len(x)
    k = config.n_neighbors
    distance = np.sqrt(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(distance, np.inf)
    from scipy.spatial.distance import cdist

    declared = cdist(x, x)
    np.fill_diagonal(declared, np.inf)
    np.testing.assert_allclose(declared, distance, rtol=1e-12, atol=1e-12)
    require(a["neighbors"].shape == (n, k), "Neighbor shape")
    dense = np.zeros((n, n))
    for i, neighbors in enumerate(a["neighbors"]):
        canonical = sorted(range(n), key=lambda j: (declared[i, j], unit_ids[j]))[:k]
        np.testing.assert_array_equal(neighbors, canonical)
        require(
            len(set(neighbors)) == k
            and i not in neighbors
            and np.all((neighbors >= 0) & (neighbors < n)),
            "Other-unit neighbor identities",
        )
        expected = sorted(range(n), key=lambda j: (distance[i, j], unit_ids[j]))[:k]
        # Independent Euclidean arithmetic can differ by an ulp at ties.
        np.testing.assert_allclose(
            np.sort(distance[i, neighbors]), np.sort(distance[i, expected]), rtol=1e-12, atol=1e-12
        )
        for j in neighbors:
            dense[i, j] += 0.5
            dense[j, i] += 0.5
    cutoff = declared[np.arange(n), a["neighbors"][:, -1]]
    ties = np.sum(declared == cutoff[:, None], axis=1)
    require(r["exact_cutoff_ties"] == ties.tolist(), "Exact distance ties")
    require(r["cutoff_tie_rows"] == int(sum(ties > 1)), "Tie-row accounting")
    from scipy.sparse import csr_matrix

    stored = csr_matrix((a["data"], a["indices"], a["indptr"]), shape=(n, n)).toarray()
    np.testing.assert_array_equal(stored, dense)
    degree = dense.sum(1)
    np.testing.assert_array_equal(a["degrees"], degree)
    comp = component_labels(dense)
    np.testing.assert_array_equal(
        comp[:, None] == comp[None, :], a["components"][:, None] == a["components"][None, :]
    )
    require(len(set(comp)) == r["components"], "Connected component count")
    lap = np.eye(n) - dense / np.sqrt(degree[:, None] * degree[None, :])
    values, vectors = np.linalg.eigh(lap)
    np.testing.assert_allclose(a["eigenvalues"], values, atol=1e-11)
    v = a["dense_eigenvectors"]
    np.testing.assert_allclose(v.T @ v, np.eye(config.max_k + 1), atol=1e-10)
    np.testing.assert_allclose(lap @ v, v * values[: config.max_k + 1], atol=1e-10)
    return lap, values, vectors, comp


def audit_embedding(a, r, lap, values, vectors, target):
    maps = a["embedding"]
    n = len(lap)
    require(maps.shape == (n, target) and np.isfinite(maps).all(), "Actual embedding")
    v = np.sqrt(a["degrees"])[:, None] * maps
    residual = float(np.linalg.norm(lap @ v - v * values[:target]))
    gram = float(np.linalg.norm(v.T @ v - np.eye(target)))
    projector = float(np.linalg.norm(v @ v.T - vectors[:, :target] @ vectors[:, :target].T))
    for name, value in (
        ("residual", residual),
        ("gram_error", gram),
        ("projector_error", projector),
    ):
        np.testing.assert_allclose(r[name], value, rtol=1e-5, atol=1e-9)
    piv = a["pivots"]
    require(sorted(piv.tolist()) == list(range(n)), "Pivot permutation")
    residual_columns = maps.T.copy()
    selected = []
    for pivot in piv[:target]:
        norms = (residual_columns**2).sum(0)
        norms[selected] = -np.inf
        require(norms[pivot] >= norms.max() - 1e-10, "Greedy QR pivot criterion")
        q = residual_columns[:, pivot] / np.linalg.norm(residual_columns[:, pivot])
        residual_columns -= q[:, None] * (q @ residual_columns)[None, :]
        selected.append(int(pivot))
    u, s, vh = np.linalg.svd(maps[piv[:target], :].T, full_matrices=False)
    rotated = np.abs(maps @ (u @ vh))
    np.testing.assert_allclose(s, a["qr_singular_values"], atol=1e-10)
    np.testing.assert_allclose(rotated, a["rotated"], atol=1e-10)
    labels = rotated.argmax(1)
    np.testing.assert_array_equal(labels, a["proposed_labels"])
    ordered = np.sort(rotated, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    np.testing.assert_allclose(margin, a["assignment_margin"], atol=1e-10)
    if residual > 1e-8 or gram > 1e-8 or projector > 1e-6:
        return "unresolved_numerical_embedding"
    if np.any(margin <= 1e-10 * np.maximum(1, ordered[:, -1])):
        return "ambiguous_qr_assignment"
    if len(set(labels)) != target:
        return "requested_k_not_obtained"
    return None


def audit_spectral(x, ids, config, r, a):
    np.testing.assert_allclose(a["raw_values"], x, atol=0, rtol=0)
    np.testing.assert_allclose(a["means"], x.mean(0), atol=1e-12)
    np.testing.assert_allclose(a["scales"], np.sqrt(np.var(x, axis=0)), atol=1e-12)
    standardized = (x - np.mean(x, axis=0)) / np.sqrt(np.var(x, axis=0))
    np.testing.assert_allclose(a["scores"], standardized, atol=1e-12)
    lap, values, vectors, comp = audit_geometry(standardized, ids, config, r, a)
    gaps = np.array(
        [a["eigenvalues"][i] - a["eigenvalues"][i - 1] for i in range(1, config.max_k + 1)]
    )
    np.testing.assert_array_equal(gaps, a["gaps"])
    ranked = sorted(range(len(gaps)), key=lambda i: (-gaps[i], i))
    k = ranked[0] + 1
    lead = float(gaps[ranked[0]] - gaps[ranked[1]])
    require(k == r["proposed_k"] and abs(lead - r["gap_lead"]) < 1e-12, "Outcome-free K choice")
    reason = None
    if gaps[ranked[0]] <= 1e-8 or lead <= 1e-8:
        reason = "unresolved_gap_choice"
    elif k == config.max_k:
        reason = "search_boundary"
    elif len(set(comp)) not in (1, k):
        reason = "unsupported_component_count"
    elif r["status"] == "failed":
        require(
            r["solver_called"] and bool(r["error"]) and "labels" not in a, "Retained solver failure"
        )
        return
    elif r["solver_called"]:
        require(len(set(comp)) == 1 and 2 <= k < config.max_k, "Solver policy")
        reason = audit_embedding(a, r, lap, values, vectors, k)
    else:
        labels = np.zeros(len(x), dtype=int) if k == 1 else comp
        require(r["method"] == ("constant" if k == 1 else "components"), "Fit-free policy")
        np.testing.assert_array_equal(
            a["proposed_labels"][:, None] == a["proposed_labels"][None, :],
            labels[:, None] == labels[None, :],
        )
    if reason is None:
        sizes = [int(sum(a["proposed_labels"] == j)) for j in sorted(set(a["proposed_labels"]))]
        require(r["cluster_sizes"] == sizes, "Proposed cluster sizes")
        if min(sizes) < config.min_cluster_size:
            reason = "small_cluster"
    require(r["reason"] == reason, "Abstention priority")
    if reason is None:
        require(r["status"] == "available" and r["selected_k"] == k, "Available selection")
        np.testing.assert_array_equal(a["labels"], a["proposed_labels"])
        require(len(a["labels"]) == len(x) and len(set(a["labels"])) == k, "Complete assignments")
    else:
        require(
            r["status"] == "withheld" and r["selected_k"] is None and "labels" not in a,
            "Withheld result cannot become K=1",
        )
