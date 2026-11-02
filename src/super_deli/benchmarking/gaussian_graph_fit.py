"""Benchmark-only plug-in Gaussian graph profiles, not a species-significance API."""

import hashlib

import numpy as np
from scipy.linalg import eigh

from super_deli.benchmarking.gaussian_graph_config import SETTINGS, draw_seed
from super_deli.benchmarking.unknown_k_data import model_config
from super_deli.models.spectral_graph import embedding, graph, propose


def digest(*values):
    h = hashlib.sha256()
    for value in values:
        a = np.ascontiguousarray(value)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def fit_gaussian(x):
    if x.ndim != 2 or len(x) < 32 or not np.isfinite(x).all():
        raise ValueError("Complete finite raw matrix with at least 32 rows required")
    mean = x.mean(0)
    delta = x - mean
    cov = delta.T @ delta / len(x)
    values, vectors = eigh(cov)
    tolerance = 64 * np.finfo(float).eps * max(x.shape) * values[-1]
    if not np.isfinite(values).all() or values[0] < -tolerance:
        raise ValueError("Unresolved covariance PSD fit")
    keep = values > tolerance
    if keep.sum() < 2:
        raise ValueError("Gaussian effective rank below two")
    return {
        "mean": mean,
        "covariance": cov,
        "eigenvalues": values,
        "eigenvectors": vectors,
        "factor": vectors[:, keep] * np.sqrt(values[keep]),
        "rank_tolerance": np.array(tolerance),
        "effective_rank": np.array(keep.sum()),
        "removed_eigenvalues": values[~keep],
    }


def standardized(x):
    mean, scale = x.mean(0), x.std(0)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid raw-feature standardization")
    scores = (x - mean) / scale
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite graph scores")
    return {"raw_values": x, "means": mean, "scales": scale, "scores": scores}


def observed_graph(x, ids, neighbors):
    a = standardized(x)
    affinity, evidence, info = graph(a["scores"], ids, model_config(f"spectral_{neighbors}"))
    a.update(evidence)
    return affinity, a, info


def tail_decision(observed, pilot, reference):
    """Frozen independent-pilot normalization; incomplete references never return a rank."""
    pcount, rcount = SETTINGS["pilot_draws"], SETTINGS["reference_draws"]
    result = {
        "status": "withheld",
        "selected_k": None,
        "p_reference": None,
        "proposed_k": None,
        "reason": None,
    }
    if observed.shape != (5,) or pilot.shape != (pcount, 5) or reference.shape != (rcount, 5):
        return {**result, "reason": "incomplete_reference_draws"}, {}
    if not all(np.isfinite(a).all() for a in (observed, pilot, reference)):
        return {**result, "reason": "nonfinite_reference_draws"}, {}
    mean, sd = pilot.mean(0), pilot.std(0, ddof=1)
    a = {"pilot_mean": mean, "pilot_sd": sd}
    if np.any(sd <= SETTINGS["sd_guard"]):
        return {**result, "reason": "degenerate_pilot_normalizer"}, a
    z = (observed - mean) / sd
    null_z = (reference - mean) / sd
    if not np.isfinite(z).all() or not np.isfinite(null_z).all():
        return {**result, "reason": "nonfinite_reference_statistic"}, a
    maxima = null_z.max(1)
    statistic = float(z.max())
    exceed = int(np.count_nonzero(maxima >= statistic))
    p = (1 + exceed) / 100
    a.update(z=z, reference_z=null_z, reference_maxima=maxima)
    result.update(
        p_reference=p,
        exceedances=exceed,
        reference_ties=int(sum(maxima == statistic)),
        statistic=statistic,
        screen_pass=p <= 0.05,
    )
    if p > 0.05:
        return {**result, "proposed_k": 1, "interpretation": "no_reference_supported_split"}, a
    ranked = np.argsort(-z, kind="stable")
    k = int(ranked[0]) + 2
    lead = float(z[ranked[0]] - z[ranked[1]])
    reason = "unresolved_gap_choice" if lead <= 1e-8 else "search_boundary" if k == 6 else None
    return {**result, "proposed_k": k, "gap_lead": lead, "reason": reason}, a


def assign(affinity, evidence, info, selection, neighbors):
    """Apply existing assignment guards to a selected K without rebuilding its graph."""
    a = {k: v.copy() for k, v in evidence.items()}
    r = {**info, **selection, "solver_called": False, "selected_k": None, "status": "withheld"}
    k = r["proposed_k"]
    try:
        if r["reason"]:
            return r, a
        if k == 1:
            # Reference-compatible K=1 remains a computational no-split decision,
            # including a fragmented graph; legacy K=1 still requires connectivity.
            if not r.get("interpretation") and info["components"] != 1:
                r["reason"] = "unsupported_component_count"
                return r, a
            a["proposed_labels"] = np.zeros(len(a["scores"]), dtype=int)
            r["method"] = "constant"
        elif info["components"] == k:
            a["proposed_labels"] = a["components"].copy()
            r["method"] = "components"
        elif info["components"] != 1:
            r["reason"] = "unsupported_component_count"
            return r, a
        else:
            r["solver_called"] = True
            e, diagnostic = embedding(affinity, a, k, 71323)
            a.update(e)
            r.update(diagnostic, method="spectral")
        if r["reason"] is None:
            sizes = [int(sum(a["proposed_labels"] == j)) for j in sorted(set(a["proposed_labels"]))]
            r["cluster_sizes"] = sizes
            if min(sizes) < model_config(f"spectral_{neighbors}").min_cluster_size:
                r["reason"] = "small_cluster"
            else:
                a["labels"] = a["proposed_labels"].copy()
                r.update(status="available", selected_k=k)
    except Exception as exc:
        r.update(status="failed", reason="solver_or_assignment_failure", error=str(exc))
    return r, a


def legacy_selection(values):
    gaps, selection = propose(values, 6)
    return selection, gaps


def reference_profiles(task, condition, fitted, ids):
    """Retain spectra/checksums/seeds; raw reference graphs are regenerated by audit."""
    profiles = {n: {"pilot": [], "reference": []} for n in (5, 15)}
    ledger = []
    for role in ("pilot", "reference"):
        for index in range(SETTINGS[role + "_draws"]):
            seed = draw_seed(task, condition, role, index)
            row = {"role": role, "index": index, "seed": seed, "graphs": {}}
            try:
                raw = (
                    fitted["mean"]
                    + np.random.default_rng(seed).normal(
                        size=(task["n"], fitted["factor"].shape[1])
                    )
                    @ fitted["factor"].T
                )
                row["raw_sha256"] = digest(raw)
                a = standardized(raw)
                for neighbors in (5, 15):
                    try:
                        _, e, info = graph(a["scores"], ids, model_config(f"spectral_{neighbors}"))
                        profiles[neighbors][role].append(e["eigenvalues"])
                        row["graphs"][str(neighbors)] = {
                            "status": "complete",
                            "components": info["components"],
                            "graph_sha256": digest(e["indptr"], e["indices"], e["data"]),
                            "spectrum_sha256": digest(e["eigenvalues"]),
                        }
                    except Exception as exc:
                        profiles[neighbors][role].append(np.full(task["n"], np.nan))
                        row["graphs"][str(neighbors)] = {"status": "failed", "error": str(exc)}
            except Exception as exc:
                row.update(status="failed", error=str(exc))
                for neighbors in (5, 15):
                    profiles[neighbors][role].append(np.full(task["n"], np.nan))
            ledger.append(row)
    arrays = {
        f"n{n}_{role}_spectra": np.array(value)
        for n, roles in profiles.items()
        for role, value in roles.items()
    }
    return ledger, arrays
