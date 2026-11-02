"""Independent Gaussian, reference-graph and tail arithmetic for retained evidence."""

import hashlib

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

from super_deli.benchmarking.neural_analysis import require


def checksum(*values):
    result = hashlib.sha256()
    for value in values:
        a = np.ascontiguousarray(value)
        result.update(str(a.dtype).encode())
        result.update(str(a.shape).encode())
        result.update(a.tobytes())
    return result.hexdigest()


def gaussian_audit(x, a):
    mean = np.sum(x, axis=0) / len(x)
    centered = x - mean
    cov = np.einsum("ni,nj->ij", centered, centered) / len(x)
    np.testing.assert_allclose(a["mean"], mean, atol=1e-12)
    np.testing.assert_allclose(a["covariance"], cov, atol=1e-11)
    values = np.linalg.eigvalsh(cov)
    tolerance = 64 * np.finfo(float).eps * max(x.shape) * values[-1]
    np.testing.assert_allclose(a["eigenvalues"], values, atol=1e-11)
    np.testing.assert_allclose(a["rank_tolerance"], tolerance, rtol=1e-10)
    kept = a["eigenvalues"] > a["rank_tolerance"]
    require(int(a["effective_rank"]) == int(kept.sum()) >= 2, "PSD effective rank")
    require(values.min() >= -tolerance, "PSD Gaussian covariance")
    v = a["eigenvectors"]
    np.testing.assert_allclose(v.T @ v, np.eye(x.shape[1]), atol=1e-10)
    np.testing.assert_allclose(cov @ v, v * a["eigenvalues"], atol=1e-10)
    np.testing.assert_allclose(a["removed_eigenvalues"], a["eigenvalues"][~kept], atol=0)
    np.testing.assert_array_equal(a["factor"], v[:, kept] * np.sqrt(a["eigenvalues"][kept]))
    np.testing.assert_allclose(
        a["factor"] @ a["factor"].T, cov, atol=2 * float(a["rank_tolerance"]), rtol=1e-12
    )


def graph_reference(raw, ids, neighbors):
    scale = np.sqrt(np.var(raw, axis=0))
    require(np.isfinite(scale).all() and np.all(scale > 0), "Reference standardization")
    standardized = (raw - np.mean(raw, axis=0)) / scale
    distances = cdist(standardized, standardized)
    # Separately evaluate Euclidean equations; do not use model graph helpers.
    euclidean = np.sqrt(
        np.einsum(
            "ijk,ijk->ij",
            standardized[:, None] - standardized[None, :],
            standardized[:, None] - standardized[None, :],
        )
    )
    np.testing.assert_allclose(distances, euclidean, atol=1e-12, rtol=1e-12)
    np.fill_diagonal(distances, np.inf)
    ranks = np.argsort(np.argsort(np.asarray(ids), kind="stable"), kind="stable")
    rank_grid = np.broadcast_to(ranks, distances.shape)
    nearest = np.lexsort((rank_grid, distances), axis=1)[:, :neighbors]
    b = np.zeros(distances.shape)
    b[np.arange(len(raw))[:, None], nearest] = 1
    adjacency = (b + b.T) / 2
    degree = adjacency.sum(axis=1)
    lap = np.eye(len(raw)) - adjacency / np.sqrt(np.outer(degree, degree))
    affinity = csr_matrix(adjacency)
    components = connected_components(affinity, directed=False)[0]
    return (
        np.linalg.eigvalsh(lap),
        checksum(affinity.indptr, affinity.indices, affinity.data),
        components,
    )


def expected_seed(task, condition, role, index):
    from super_deli.benchmarking.unknown_k_config import CASES

    return [
        task["draw_root"],
        list(CASES).index(task["case"]),
        task["n"],
        task["replicate"],
        ["clean", "noisy"].index(condition),
        ["pilot", "reference"].index(role),
        index,
    ]


def audit_profiles(task, condition, ids, fitted, ledger, spectra):
    require(
        set(spectra)
        == {f"n{n}_{role}_spectra" for n in (5, 15) for role in ("pilot", "reference")},
        "Reference spectrum schema",
    )
    require(
        all(a.shape == (49 if "pilot" in key else 99, task["n"]) for key, a in spectra.items()),
        "Reference spectrum dimensions",
    )
    expected_keys = {
        (role, i) for role, count in (("pilot", 49), ("reference", 99)) for i in range(count)
    }
    require(
        len(ledger) == 148 and {(r["role"], r["index"]) for r in ledger} == expected_keys,
        "Complete pilot/reference schedule without replacements",
    )
    for row in ledger:
        role, i = row["role"], row["index"]
        seed = expected_seed(task, condition, role, i)
        require(row["seed"] == seed, "Disjoint source-bound pilot/reference seed")
        raw = (
            fitted["mean"]
            + np.random.default_rng(seed).standard_normal(
                (task["n"], int(fitted["effective_rank"]))
            )
            @ fitted["factor"].T
        )
        if row.get("status") == "failed":
            require(bool(row["error"]) and not row["graphs"], "Retained raw-draw failure")
            require(
                all(np.isnan(spectra[f"n{n}_{role}_spectra"][i]).all() for n in (5, 15)),
                "Failed raw draw has no completed reference spectra",
            )
            if "raw_sha256" in row:
                require(row["raw_sha256"] == checksum(raw), "Retained failed raw draw hash")
            continue
        require(row["raw_sha256"] == checksum(raw), "Exact regenerated raw reference")
        for neighbors in (5, 15):
            stored = spectra[f"n{neighbors}_{role}_spectra"][i]
            record = row["graphs"][str(neighbors)]
            if record["status"] == "failed":
                require(bool(record["error"]) and np.isnan(stored).all(), "Retained failed draw")
                continue
            values, graph_hash, comp = graph_reference(raw, ids, neighbors)
            require(record["graph_sha256"] == graph_hash, "Independently regenerated graph hash")
            require(record["spectrum_sha256"] == checksum(stored), "Stored spectrum hash")
            require(record["components"] == comp, "Reference components")
            np.testing.assert_allclose(values, stored, atol=1e-11, rtol=1e-11)


def tail_reference(values, pilot_spectra, reference_spectra):
    pilot = pilot_spectra[:, 2:7] - pilot_spectra[:, 1:6]
    draws = reference_spectra[:, 2:7] - reference_spectra[:, 1:6]
    if pilot.shape != (49, 5) or draws.shape != (99, 5):
        return {"reason": "incomplete_reference_draws", "p_reference": None}
    if not np.isfinite(pilot).all() or not np.isfinite(draws).all():
        return {"reason": "nonfinite_reference_draws", "p_reference": None}
    mean = pilot.sum(axis=0) / 49
    sd = np.sqrt(((pilot - mean) ** 2).sum(axis=0) / 48)
    if (sd <= 1e-12).any():
        return {"reason": "degenerate_pilot_normalizer", "p_reference": None}
    z = ((values[2:7] - values[1:6]) - mean) / sd
    null_z = (draws - mean) / sd
    if not np.isfinite(z).all() or not np.isfinite(null_z).all():
        return {"reason": "nonfinite_reference_statistic", "p_reference": None}
    ref_maxima = np.max(null_z, axis=1)
    exceed = int(sum(ref_maxima >= np.max(z)))
    p = (exceed + 1) / 100
    ranked = sorted(range(5), key=lambda k: (-z[k], k))
    k = 1 if p > 0.05 else ranked[0] + 2
    reason = None
    if k != 1:
        if z[ranked[0]] - z[ranked[1]] <= 1e-8:
            reason = "unresolved_gap_choice"
        elif k == 6:
            reason = "search_boundary"
    return {
        "reason": reason,
        "p_reference": p,
        "proposed_k": k,
        "exceedances": exceed,
        "z": z,
        "pilot_mean": mean,
        "pilot_sd": sd,
        "reference_maxima": ref_maxima,
    }
