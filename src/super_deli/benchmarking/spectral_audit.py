"""Read-only distance, Laplacian, eigenspace, assignment and replay audits."""

import time

import numpy as np

from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_moons_archive import inventory
from super_deli.benchmarking.spectral_archive import load_coordinates
from super_deli.benchmarking.spectral_config import density_jobs, graph_jobs
from super_deli.benchmarking.spectral_study import check_frozen, job_path
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def component_labels(a):
    n = len(a)
    labels = np.full(n, -1, dtype=int)
    c = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        labels[i] = c
        queue = [i]
        while queue:
            j = queue.pop()
            for k in np.flatnonzero(a[j] > 0):
                if labels[k] < 0:
                    labels[k] = c
                    queue.append(int(k))
        c += 1
    return labels


def audit_graph(x, unit_ids, job, r, a):
    if not a:
        require(
            bool(r.get("error")) and all(p["status"] == "failed" for p in r["partitions"]),
            "Retained graph failure",
        )
        return
    n = len(x)
    k = job["neighbors"]
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
    np.testing.assert_allclose(v.T @ v, np.eye(5), atol=1e-10)
    np.testing.assert_allclose(lap @ v, v * values[:5], atol=1e-10)
    require([p["target_k"] for p in r["partitions"]] == [2, 4], "Supplied-K accounting")
    for record in r["partitions"]:
        target = record["target_k"]
        p = {
            key[len(f"K{target}_") :]: value
            for key, value in a.items()
            if key.startswith(f"K{target}_")
        }
        if r["components"] == target:
            require(
                record["status"] == "available"
                and record["method"] == "components"
                and not record["solver_called"],
                "Direct component policy",
            )
            np.testing.assert_array_equal(
                p["labels"][:, None] == p["labels"][None, :], comp[:, None] == comp[None, :]
            )
        elif r["components"] != 1:
            require(
                record["status"] == "withheld"
                and record["reason"] == "unsupported_component_count"
                and not p,
                "Disconnected policy",
            )
        elif values[target] - values[target - 1] <= 1e-8:
            require(
                record["status"] == "withheld"
                and record["reason"] == "unresolved_cutoff_eigenspace"
                and not p,
                "Cutoff gap policy",
            )
        elif record["status"] == "failed":
            require(bool(record["error"]) and record["solver_called"], "Retained solver failure")
        else:
            maps = p["embedding"]
            require(
                maps.shape == (n, target) and np.isfinite(maps).all(),
                "Embedding shape and finiteness",
            )
            v = np.sqrt(degree)[:, None] * maps
            residual = float(np.linalg.norm(lap @ v - v * values[:target]))
            gram = float(np.linalg.norm(v.T @ v - np.eye(target)))
            projector = float(np.linalg.norm(v @ v.T - vectors[:, :target] @ vectors[:, :target].T))
            for name, value in (
                ("residual", residual),
                ("gram_error", gram),
                ("projector_error", projector),
            ):
                np.testing.assert_allclose(record[name], value, rtol=1e-5, atol=1e-9)
            piv = p["pivots"]
            require(sorted(piv.tolist()) == list(range(n)), "QR pivot permutation")
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
            np.testing.assert_allclose(s, p["qr_singular_values"], atol=1e-10)
            np.testing.assert_allclose(rotated, p["rotated"], atol=1e-10)
            labels = rotated.argmax(1)
            np.testing.assert_array_equal(p["proposed_labels"], labels)
            ordered = np.sort(rotated, axis=1)
            margin = ordered[:, -1] - ordered[:, -2]
            np.testing.assert_allclose(p["assignment_margin"], margin, atol=1e-10)
            reason = (
                "unresolved_numerical_embedding"
                if residual > 1e-8 or gram > 1e-8 or projector > 1e-6
                else "ambiguous_qr_assignment"
                if np.any(margin <= 1e-10 * np.maximum(1, ordered[:, -1]))
                else "requested_k_not_obtained"
                if len(set(labels)) != target
                else None
            )
            if reason:
                require(
                    record["status"] == "withheld"
                    and record["reason"] == reason
                    and "labels" not in p,
                    "Numerical/assignment policy",
                )
            else:
                require(
                    record["status"] == "available" and record["method"] == "spectral",
                    "Available partition policy",
                )
                np.testing.assert_array_equal(p["labels"], labels)
        if record["status"] == "available":
            require(
                len(p["labels"]) == n
                and len(set(p["labels"])) == target
                and np.all(p["labels"] >= 0),
                "All-assigned supplied count",
            )
            sizes = [int(sum(p["labels"] == j)) for j in set(p["labels"])]
            require(
                record["cluster_sizes"] == sizes
                and record["small_clusters"] == sum(s < 5 for s in sizes),
                "Small-cluster accounting",
            )


def audit_density(record, a, n):
    if record["status"] == "failed":
        require(bool(record["error"]), "Retained density failure")
        return
    require(a["labels"].shape == a["strength"].shape == (n,), "Density alignment")
    require(a["labels"].dtype.kind in "iu" and np.all(a["labels"] >= -1), "Density noise label")
    require(
        np.isfinite(a["strength"]).all() and np.all((a["strength"] >= 0) & (a["strength"] <= 1)),
        "Density strengths",
    )
    require(np.all(a["strength"][a["labels"] < 0] == 0), "Noise strengths zero")
    require(len(set(a["labels"]) - {-1}) == record["selected_k"], "Density selected K")
    require(
        all(sum(a["labels"] == k) >= 5 for k in set(a["labels"]) - {-1}), "Density minimum size"
    )


def audit_phase(root, phase):
    start = time.monotonic()
    frozen, _ = check_frozen(root)
    binding = read_json(root / "binding.json")
    require(
        inventory(binding["source_root"]) == binding["source_sha256"],
        "Full source preservation inventory",
    )
    actual = {
        str(p.relative_to(root / phase)): fingerprint_file(p).sha256
        for p in (root / phase).rglob("*")
        if p.is_file()
    }
    require(actual == read_json(root / f"{phase}-inventory.json"), "Complete result inventory")
    lookup = {e["key"]: e for e in binding["observations"]}
    totals = {
        "evaluations": 0,
        "available": 0,
        "withheld": 0,
        "failed": 0,
        "solver_called": 0,
        "component_partitions": 0,
        "exact_graph_replays": 0,
        "density_replays": 0,
    }
    for job in graph_jobs(binding, phase):
        out = job_path(root, phase, job)
        r, a = read_json(out / "result.json"), arrays(out / "arrays.npz")
        require(r["job"] == job, "Frozen graph job")
        e = lookup[job["key"]]
        audit_graph(load_coordinates(root, e)[job["space"]], e["unit_ids"], job, r, a)
        for p in r["partitions"]:
            totals["evaluations"] += 1
            totals[p["status"]] += 1
            totals["solver_called"] += p["solver_called"]
            totals["component_partitions"] += p.get("method") == "components"
        if job["kind"] == "replay":
            original = job_path(root, phase, {**job, "kind": "point"})
            old = arrays(original / "arrays.npz")
            require(set(old) == set(a), "Graph replay inventory")
            for key, value in a.items():
                np.testing.assert_array_equal(value, old[key])
            before = read_json(original / "result.json")
            for after, initial in zip(r["partitions"], before["partitions"], strict=True):
                require(
                    {k: v for k, v in after.items() if k != "seconds"}
                    == {k: v for k, v in initial.items() if k != "seconds"},
                    "Replay status/diagnostics",
                )
            totals["exact_graph_replays"] += 1
    for job in density_jobs(binding, phase):
        out = job_path(root, phase, job, True)
        r, a = read_json(out / "result.json"), arrays(out / "arrays.npz")
        e = lookup[job["key"]]
        audit_density(r, a, e["n"])
        totals["evaluations"] += 1
        if r["status"] == "complete":
            coords = load_coordinates(root, e)
            np.testing.assert_array_equal(a["scores"], coords[job["space"]])
            if job["kind"] in ("replay", "archive-replay"):
                old = (
                    arrays(job_path(root, phase, {**job, "kind": "point"}, True) / "arrays.npz")
                    if job["kind"] == "replay"
                    else {
                        "labels": coords[job["space"] + "_labels"],
                        "strength": coords[job["space"] + "_strength"],
                    }
                )
                for key in ("labels", "strength"):
                    np.testing.assert_array_equal(a[key], old[key])
                totals["density_replays"] += 1
        else:
            totals["failed"] += 1
    require(totals["evaluations"] == frozen["counts"][phase], "Frozen evaluation total")
    result = {
        "passed": True,
        "phase": phase,
        **totals,
        "seconds": time.monotonic() - start,
        "source_observations": len(binding["observations"]),
        "neural_training_fits": 0,
        "audit_boundary": (
            "independent graph/eigenspace/QR equations; "
            "no second complete library eigensolver or HDBSCAN"
        ),
    }
    write_json(result, root / f"{phase}-audit.json")
    return result
